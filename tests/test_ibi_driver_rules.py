"""The driver's per-coordinate rules, and the default path they must not perturb.

Every table in results/ibi_relax was produced by ibi_loop's inline update dispatch. The two-lever arm
adds per-coordinate rules to that dispatch, so the first thing these tests do is pin the DEFAULT path
against golden digests captured before the rules existed:

    plan_update        5a65353e88b5acd9   (max|dU| 3.884586660511, U sum 389.497483526829)
    moment_correction  1ece619997c290b5   (max|dU| 1.350406900727, moment norm 0.129534166667)

on a deterministic synthetic input (120 bins, 39811 observations, 137 outside; counts digest
d182960bf4659f1a, p_ref digest ab334d562a5ea196). A refactor that moved the dispatch into a function
and a rule system that shares it both have to leave those numbers alone.
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B      # noqa: E402
import ibi_bonded as I            # noqa: E402
import ibi_loop as L              # noqa: E402
import plan_c_basis as PB         # noqa: E402

GOLDEN = {"plan_update": {"max_abs_dU": 3.884586660511, "U_sum": 389.497483526829,
                          "digest": "5a65353e88b5acd9"},
          "moment_correction": {"max_abs_dU": 1.350406900727, "moment_norm": 0.129534166667,
                                "U_sum": 196.333217305197, "digest": "1ece619997c290b5"}}


def _digest(u):
    return hashlib.sha1(np.asarray(u, dtype=float).tobytes()).hexdigest()[:16]


def _inputs():
    """The synthetic input the golden values were captured on, rebuilt identically."""
    rng = np.random.default_rng(20261001)
    nb = 120
    edges = np.linspace(-1.0, 1.0, nb + 1)
    centre = 0.5 * (edges[:-1] + edges[1:])
    table = {"lo": -1.0, "hi": 1.0, "binw": 2.0 / nb, "centre": centre,
             "U": 6.0 * centre ** 2, "sigma": 0.2}
    counts, _ = np.histogram(rng.normal(0.15, 0.35, size=40_000), bins=edges)
    counts = counts.astype(np.int64)
    ref_counts, _ = np.histogram(rng.normal(0.10, 0.30, size=40_000), bins=edges)
    p_ref = I.probability_from_counts(ref_counts.astype(np.int64))
    n_out = 137
    n_tot = int(counts.sum()) + n_out
    hist = I.SimHistogram(counts=np.asarray(counts, float), n=n_tot, n_outside=n_out,
                          lo=-1.0, hi=1.0, nbins=nb)
    return table, counts, n_tot, n_out, p_ref, hist


def test_default_table_path_is_bit_identical_to_before():
    """IBI_LOOP_OPERATOR unset: update_one_coord IS the old inline plan_update call."""
    table, counts, n_tot, n_out, p_ref, hist = _inputs()
    assert L.OPERATOR == "table", "the environment leaked an operator into the test"
    res, rule = L.update_one_coord("angle", table, counts, n_tot, n_out, p_ref, hist,
                                   {c: [] for c in L.UPDATED}, {c: [] for c in L.UPDATED})
    U = np.asarray(res.require_table()["U"], dtype=float)
    assert res.status == "ok"
    assert float(res.max_abs_dU) == pytest.approx(GOLDEN["plan_update"]["max_abs_dU"], abs=1e-12)
    assert float(U.sum()) == pytest.approx(GOLDEN["plan_update"]["U_sum"], abs=1e-9)
    assert _digest(U) == GOLDEN["plan_update"]["digest"]
    assert rule == "table"


def test_default_moment_path_is_bit_identical_to_before(monkeypatch):
    """IBI_LOOP_OPERATOR=moments: same digest, same moment norm."""
    monkeypatch.setattr(L, "OPERATOR", "moments")
    monkeypatch.setattr(L, "GAIN_BY_COORD", {c: 1.0 for c in L.UPDATED})
    table, counts, n_tot, n_out, p_ref, hist = _inputs()
    res, rule = L.update_one_coord("angle", table, counts, n_tot, n_out, p_ref, hist,
                                   {c: [] for c in L.UPDATED}, {c: [] for c in L.UPDATED})
    U = np.asarray(res.require_table()["U"], dtype=float)
    assert float(res.max_abs_dU) == pytest.approx(GOLDEN["moment_correction"]["max_abs_dU"], abs=1e-12)
    assert float(res.diagnostics["moment_norm"]) == pytest.approx(
        GOLDEN["moment_correction"]["moment_norm"], abs=1e-12)
    assert float(U.sum()) == pytest.approx(GOLDEN["moment_correction"]["U_sum"], abs=1e-9)
    assert _digest(U) == GOLDEN["moment_correction"]["digest"]
    assert rule == "moments"


def test_bspline16_rule_is_a_different_step_and_uses_the_big_basis():
    """The dihedral's rule in the two-lever arm: the moment operator on a 16-function B-spline."""
    table, counts, n_tot, n_out, p_ref, hist = _inputs()
    res, rule = L.update_one_coord("dihedral", table, counts, n_tot, n_out, p_ref, hist,
                                   {c: [] for c in L.UPDATED}, {c: [] for c in L.UPDATED},
                                   rule="bspline16")
    assert rule == "bspline16"
    U = np.asarray(res.require_table()["U"], dtype=float)
    assert np.isfinite(U).all()
    assert res.diagnostics["K"] == L.RULE_BSPLINE_M == 16
    assert _digest(U) != GOLDEN["moment_correction"]["digest"], "the rule did nothing"
    assert np.abs(U - np.asarray(table["U"], float)).max() > 0


def test_bspline16_rule_passes_the_eigenvalue_relative_ridge(monkeypatch):
    """What the rule actually hands the operator. The ridge FORM is the measured lesson
    (docs/archive/plan_c_basis_family.md 2.4): trace/m falls with m while eig_max does not, so the same
    ridge_rel regularises 8.8x less at the fine end and the trace form stops working there."""
    seen = {}
    real = I.moment_correction

    def spy(table, counts, n, n_outside, p_ref, **kw):
        seen.update(kw)
        return real(table, counts, n, n_outside, p_ref, **kw)

    monkeypatch.setattr(L.I, "moment_correction", spy)
    table, counts, n_tot, n_out, p_ref, hist = _inputs()
    L.update_one_coord("dihedral", table, counts, n_tot, n_out, p_ref, hist,
                       {c: [] for c in L.UPDATED}, {c: [] for c in L.UPDATED}, rule="bspline16")
    assert seen["ridge_form"] == "eig"
    assert seen["ridge_rel"] == L.RULE_RIDGE_REL
    assert seen["design"].shape[1] == L.RULE_BSPLINE_M == 16
    assert seen["gain"] == L.GAIN_BY_COORD["dihedral"]


def test_trace_and_eigenvalue_ridges_differ_on_this_basis():
    """The premise of the form above, measured rather than assumed: on a B-spline design of the size
    the arm uses, trace(cov)/n and eig_max(cov) are far apart."""
    table, counts, n_tot, n_out, p_ref, hist = _inputs()
    A = PB.design_bspline(table["centre"], table["lo"], table["hi"], 16)
    w_sim = counts / counts.sum()
    sim = w_sim @ A
    cov = (w_sim[:, None] * A).T @ A - np.outer(sim, sim)
    trace_form = float(np.trace(cov)) / cov.shape[0]
    eig_form = float(np.linalg.eigvalsh(cov)[-1])
    assert eig_form > 2.0 * trace_form, (trace_form, eig_form)


def test_selfconsistent_rule_is_a_replacement_that_ignores_p_ref():
    """The angle's rule in the two-lever arm. Two properties matter and both are tested:

    1. it is plan_c_basis.fit of the ensemble's own -kBT ln p on the Chebyshev K=8 design, with the
       C2s pipeline's ridge -- the same rule the seven-chain C2s8/CA16 arms ran, so the full-pool arm
       is comparable with them;
    2. it does not read p_ref at all, which is what makes it compatible with the loop's invariant that
       the reference is fixed (an INCREMENT's target): an increment against a self-consistent target is
       identically zero, so this has to be a replacement fit.
    """
    table, counts, n_tot, n_out, p_ref, hist = _inputs()
    empty = {c: [] for c in L.UPDATED}
    res, rule = L.update_one_coord("angle", table, counts, n_tot, n_out, p_ref, hist, empty, empty,
                                   rule="selfconsistent")
    assert rule == "selfconsistent"
    p_ens = I.probability_from_counts(counts.astype(float))
    target = -B.KBT * np.log(p_ens)
    target = target - target.min()
    A, _x = I._chebyshev_design(table["centre"], table["lo"], table["hi"], L.CORRECTION_K)
    U_want, _d = PB.fit(A, table, counts, U_target=target, ridge_rel=L.RULE_C2S_RIDGE_REL,
                        ridge_form="trace", gain=L.GAIN_BY_COORD["angle"],
                        support_frac=L.RULE_C2S_SUPPORT_FRAC,
                        taper_decades=L.RULE_C2S_TAPER_DECADES)
    assert np.allclose(np.asarray(res.require_table()["U"], float), np.asarray(U_want, float),
                       atol=1e-12)
    other = np.asarray(p_ref, float)[::-1].copy()
    res2, _r = L.update_one_coord("angle", table, counts, n_tot, n_out, other, hist,
                                  {c: [] for c in L.UPDATED}, {c: [] for c in L.UPDATED},
                                  rule="selfconsistent")
    assert np.allclose(np.asarray(res2.require_table()["U"], float),
                       np.asarray(res.require_table()["U"], float), atol=1e-12), \
        "the self-consistent rule consumed p_ref"


def test_unknown_rule_falls_back_to_the_default_path():
    """A typo in an environment variable must not silently invent a rule."""
    table, counts, n_tot, n_out, p_ref, hist = _inputs()
    empty = {c: [] for c in L.UPDATED}
    res, rule = L.update_one_coord("angle", table, counts, n_tot, n_out, p_ref, hist, empty, empty,
                                   rule="bspline99")
    assert rule == "table"
    assert _digest(np.asarray(res.require_table()["U"], float)) == GOLDEN["plan_update"]["digest"]

# --------------------------------------------------------------- the J scope (2026-10-01)
def test_new_default_has_no_frozen_coordinate():
    """The default path's new shape: nothing frozen, and J averaged over exactly UPDATED."""
    assert L.FROZEN == ()
    assert L.CONTROLLED == L.UPDATED
    assert L.UNCONTROLLED == ("stack",)
    assert "stack" not in L.UPDATED, "stack must not be an updated coordinate"


def test_j_leaves_out_the_derived_coordinate_and_j_all_does_not():
    """joint_J averages the controlled coordinates; joint_J_all keeps the old four-coordinate mean.

    stack is a DERIVED quantity (the P(i)-P(i+2) identity, see ibi_loop.UNCONTROLLED), so its ratio is
    bb_bond's and angle's error propagated, and a constant offset in it is something no update can
    remove. This pins that the two means differ when stack is off, and that the old one is unchanged.
    """
    import ibi_core as IC
    nb = 40
    tab = {}
    acc = {}
    for i, c in enumerate(["bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack"]):
        tab[c] = {"sigma": 0.1 + 0.01 * i}
        # (sum, sum of squares, n): a distribution 20 per cent wider than the reference for stack,
        # 2 per cent for the three controlled ones.
        wid = 1.20 if c == "stack" else 1.02
        mean, sig = 0.0, (0.1 + 0.01 * i) * wid
        n = 10000
        acc[c] = [mean * n, (sig ** 2 + mean ** 2) * n, n]
    _v, j_controlled = IC.simref(acc, tab, skip=("intra_pc", "intra_cn"),
                                 only=("bb_bond", "angle", "dihedral"))
    _v2, j_all = IC.simref(acc, tab, skip=("intra_pc", "intra_cn"))
    assert j_controlled < j_all, "the derived coordinate did not inflate the old mean"
    assert j_all == pytest.approx((abs(np.log(1.02)) * 3 + abs(np.log(1.20))) / 4, rel=1e-9)
    assert j_controlled == pytest.approx(abs(np.log(1.02)), rel=1e-9)


def _run_with_env(env_extra):
    """Import ibi_loop in a fresh interpreter with these env vars; return (rc, stdout, stderr)."""
    import subprocess
    # The child needs scripts/ and src/ on its path (the parent's sys.path.insert does not cross a
    # process boundary) and the mounted structure database, exactly as the launcher sets them.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "PYTHONPATH": os.pathsep.join([str(REPO / "scripts"), str(REPO / "src")]),
           "TORUSFOLD_RSRNASP": str(REPO / "_cgdata" / "combined"), **env_extra}
    p = subprocess.run([sys.executable, "-c",
                        "import ibi_loop as L; print('FROZEN', L.FROZEN); "
                        "print('CONTROLLED', L.CONTROLLED)"],
                       capture_output=True, text=True, env=env, cwd=str(REPO))
    return p.returncode, p.stdout, p.stderr


def test_freeze_env_parses_and_narrows_the_controlled_set():
    rc, out, err = _run_with_env({"IBI_LOOP_FREEZE": "angle"})
    assert rc == 0, err
    assert "FROZEN ('angle',)" in out
    assert "CONTROLLED ('bb_bond', 'dihedral')" in out


def test_freeze_env_rejects_a_name_that_is_not_updated():
    """A typo must stop the run: silently freezing nothing (or something not updatable) would make the
    arm's record claim a control it never ran."""
    rc, out, err = _run_with_env({"IBI_LOOP_FREEZE": "stack"})
    assert rc != 0
    assert "not updated coordinates" in (err + out)
    assert "bb_bond" in (err + out), "the legal names must be listed"

