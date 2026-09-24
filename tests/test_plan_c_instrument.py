"""What the clean instrument must do, checked without touching a sampler.

plan_c_instrument.py answers "did these two SAMPLED ensembles move?" -- the question the implied
-distribution proxy could not answer because the sampled Hamiltonian carries a wall and the chain's
coupling (docs/plan_c_c2_stabilization.md 3.4). These tests pin the pieces that a silent change
would corrupt while every number downstream still looked plausible:

  * the normalisation is ibi_bonded's pseudocount policy, NOT counts/counts.sum(). Measured on the
    real C2s8 bb_bond pair: getting that wrong reads 2.684 where the loop's own stationarity reads
    1.129, a factor of 2.4 invented out of the bins neither ensemble visited.
  * ln_mean here IS the loop's stationarity. The stored ensembles plus the round json are a golden
    value: if the two ever disagree, one of them changed meaning.
  * TV is 0 for an ensemble against itself and 1 for disjoint mass.
  * the moment and quantile columns are in units of the coordinate's own spread, so a shift of one
    known number of bins is a known number of sigmas.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ibi_bonded as I          # noqa: E402
import plan_c_instrument as PI  # noqa: E402
import plan_c_loop as L         # noqa: E402

PC = REPO / "results" / "plan_c"


def _gauss(nbins=121, centre_bin=60, width=8.0, n=200_000, seed=7):
    """A histogram on a unit grid: enough counts that the Poisson floor is far below the features."""
    rng = np.random.default_rng(seed)
    counts = rng.normal(centre_bin, width, size=n)
    h, _ = np.histogram(counts, bins=np.arange(nbins + 1) - 0.5)
    return h.astype(float)


def test_tv_is_zero_against_itself_and_near_one_for_disjoint_mass():
    """TV is bounded by 1 and the pseudocount keeps a "disjoint" pair just below it: 1000 counts in
    a bin the other ensemble never visits still carries 0.5/(1050) of probability in 50 bins."""
    h = _gauss()
    m = PI.metrics(h, h, np.arange(len(h)), reps=0)
    assert m["tv"] == pytest.approx(0.0)
    assert m["ln_mean"] == pytest.approx(0.0, abs=1e-9)
    other = np.zeros_like(h)
    other[-1] = 1000.0
    d = PI.metrics(h, other, np.arange(len(h)), reps=0)
    assert 0.9 < d["tv"] < 1.0


def test_pseudocount_policy_is_what_makes_the_ratio_finite():
    """The regression test for the 2.4x: two histograms with bins only one side visited.

    Without the pseudocount the unvisited side is p = 0, clip(1e-300) turns the ratio into about
    690, and those bins carry enough weight to move the mean. With it, the same pair reads the ln of
    the two totals.
    """
    a = _gauss(n=200_000, seed=3)
    b = _gauss(n=200_000, seed=5)
    a[100:120] += 25.0               # twenty bins only the a side ever visits, as real tails are
    good = PI.metrics(a, b, np.arange(len(a)), reps=0)
    assert good["ln_mean"] < 0.1, "the pseudocount policy did not bound the unvisited bins"
    # the wrong convention, spelled out: count normalisation with a floor at 1e-300
    pa, pb = a / a.sum(), b / b.sum()
    lr = np.abs(np.log(np.clip(pb, 1e-300, None) / np.clip(pa, 1e-300, None)))
    w = 0.5 * (pa + pb)
    poisoned = float((w * lr).sum() / w.sum())
    assert poisoned > 5.0 * good["ln_mean"], (
        f"the poisoned reading ({poisoned:.3f}) is not what this test guards")


def test_moment_and_quantile_columns_are_in_sigma_units():
    h = _gauss()
    shifted = _gauss(centre_bin=68, seed=11)          # 8 bins = 1 sigma by construction
    m = PI.metrics(h, shifted, np.arange(len(h)), reps=0)
    assert m["dmean_sig"] == pytest.approx(1.0, abs=0.05)
    assert m["dq50_sig"] == pytest.approx(1.0, abs=0.05)
    assert abs(m["dstd_sig"]) < 0.05
    one_bin = PI.metrics(h, _gauss(centre_bin=61, seed=13), np.arange(len(h)), reps=0)
    assert one_bin["dq50_sig"] == pytest.approx(0.125, abs=0.03)


def test_metrics_refuse_mismatched_grids_and_empty_ensembles():
    h = _gauss()
    with pytest.raises(ValueError):
        PI.metrics(h, h[:-1], np.arange(len(h)), reps=0)
    with pytest.raises(ValueError):
        PI.metrics(np.zeros_like(h), h, np.arange(len(h)), reps=0)


def test_poisson_floor_is_reported_and_is_below_the_feature():
    """The floor is the number a single comparison is read against, and it has to be much smaller
    than the sigma-scale shift this test puts in, or the instrument cannot see anything."""
    h = _gauss()
    m = PI.metrics(h, _gauss(centre_bin=68, seed=11), np.arange(len(h)), reps=60)
    assert "poisson_p95" in m
    assert m["poisson_p95"]["ln_mean"] < 0.2 * m["ln_mean"]
    assert m["poisson_p95"]["dq50_sig"] < 0.1 * abs(m["dq50_sig"])


def test_trend_and_opposite_read_the_series_they_are_given():
    rows = [{"tag": "t", "arm": "A", "coord": "angle", "ra": 1, "ln_mean": 0.9,
             "polluted_ln_median": 0.2},
            {"tag": "t", "arm": "A", "coord": "angle", "ra": 2, "ln_mean": 0.1,
             "polluted_ln_median": 0.8}]
    tc, tp, opp = PI.opposite(rows, "A", "angle", "t")
    assert (tc, tp, opp) == (-1, 1, True)
    assert PI.trend([1.0, None, 3.0]) == 1
    assert PI.trend([5.0]) == 0


def test_floor_rows_keep_only_the_small_steps():
    rows = [{"coord": "bb_bond", "step_std": 0.2, "ln_mean": 0.05, "tv": 0.02, "tag": "t",
             "arm": "A", "ra": 1, "rb": 2},
            {"coord": "bb_bond", "step_std": 4.0, "ln_mean": 9.0, "tv": 0.9, "tag": "t",
             "arm": "A", "ra": 2, "rb": 3}]
    fl = PI.floor_rows(rows, 1.0)
    assert fl["bb_bond"]["n"] == 1
    assert fl["bb_bond"]["ln_mean_min"] == pytest.approx(0.05)
    assert fl["bb_bond"]["tv_min"] == pytest.approx(0.02)


# --------------------------------------------------------------- golden value, if the run is here
def test_instrument_reproduces_the_loops_own_stationarity():
    """ln_mean on the stored ensembles IS plan_c_loop._stationarity -- the two are the same
    definition, and this is the check that they have not drifted apart.

    Skipped when results/plan_c is not present (it is an untracked results directory, not part of
    the checkout).
    """
    ens_p, doc_p = PC / "ensembles_c2stab.npz", PC / "plan_c_c2stab.json"
    if not (ens_p.exists() and doc_p.exists()):
        pytest.skip("results/plan_c is not present in this checkout")
    z = np.load(ens_p)
    doc = json.load(open(doc_p))
    for c in ("bb_bond", "angle", "dihedral"):
        a = z[f"C2s8_r1__{c}"].astype(float)
        b = z[f"C2s8_r2__{c}"].astype(float)
        got = PI.metrics(a, b, np.arange(len(a)), reps=0)["ln_mean"]
        want = doc["arms"]["C2s8"]["rounds"][1]["stationarity"][c]["mass_weighted_mean"]
        assert got == pytest.approx(want, rel=1e-9)


def test_step_std_reproduces_the_recorded_applied_step():
    """The applied-step column has to be recomputable from the field files alone."""
    fdir = PC / "fields" / "c2stab"
    ens_p, doc_p = PC / "ensembles_c2stab.npz", PC / "plan_c_c2stab.json"
    if not (fdir.exists() and ens_p.exists() and doc_p.exists()):
        pytest.skip("results/plan_c is not present in this checkout")
    z = np.load(ens_p)
    doc = json.load(open(doc_p))
    for rnd in (2, 3):
        fa = L.load_field(fdir / f"C2s8_r{rnd - 1}.npz")
        fb = L.load_field(fdir / f"C2s8_r{rnd}.npz")
        for c in ("bb_bond", "angle", "dihedral"):
            got = PI.step_std(fa, fb, z[f"C2s8_r{rnd}__{c}"], c)
            want = doc["arms"]["C2s8"]["rounds"][rnd - 1]["fit"][c]["mass_weighted_std_dU"]
            assert got == pytest.approx(want, rel=1e-9)

# --------------------------------------------------------------- the decomposition (item 2)
def test_load_ensembles_skips_the_decomposition(tmp_path):
    """A caller that asked for the pooled ensemble must not silently get a decomposed one."""
    p = tmp_path / "e.npz"
    np.savez(p, **{"A_r1__bb_bond": np.ones(4), "A_r1__chain0__bb_bond": np.full(4, 3.0),
                   "A_r1__blocks__bb_bond": np.ones((2, 4)),
                   "A_r1__n_total__bb_bond": np.array(4)})
    ens = PI.load_ensembles(p)
    assert set(ens["A_r1"]) == {"bb_bond"}


def test_chains_of_and_blocks_of_read_the_decomposition(tmp_path):
    """<name>_r<round>__chain<i>__<coord> and __blocks__<coord>, and nothing else."""
    p = tmp_path / "e.npz"
    blocks = np.arange(8, dtype=float).reshape(2, 4)
    np.savez(p, **{"A_r1__bb_bond": np.ones(4), "A_r1__chain0__bb_bond": np.full(4, 3.0),
                   "A_r1__chain1__bb_bond": np.full(4, 5.0),
                   "A_r1__chain0_blocks__bb_bond": np.ones((2, 4)),
                   "A_r1__blocks__bb_bond": blocks})
    z = np.load(p)
    ch = PI.chains_of(z, "A_r1", "bb_bond")
    assert sorted(ch) == [0, 1], "the per-chain-per-block key was read as a chain histogram"
    assert ch[1][0] == pytest.approx(5.0)
    assert np.allclose(PI.blocks_of(z, "A_r1", "bb_bond"), blocks)
    assert PI.blocks_of(z, "A_r2", "bb_bond") is None


def test_jackknife_matches_the_hand_computed_variance():
    v = [1.0, 2.0, 3.0, 4.0]
    n, mean = 4, 2.5
    want = np.sqrt((n - 1) / n * sum((x - mean) ** 2 for x in v))
    j = PI.jackknife(v)
    assert j["n"] == 4 and j["mean"] == pytest.approx(mean)
    assert j["std_jackknife"] == pytest.approx(want)
    assert PI.jackknife([2.0, 2.0, 2.0, 2.0])["std_jackknife"] == pytest.approx(0.0)
    assert PI.jackknife([1.0])["n"] == 1 and np.isnan(PI.jackknife([1.0])["std_jackknife"])
    assert np.isnan(PI.jackknife([])["std_jackknife"])

