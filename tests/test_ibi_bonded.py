"""The IBI update on a case whose answer is known, and the refusals it must produce.

scripts/ibi_bonded.py is the update half of iterative Boltzmann inversion: given the table
that was simulated, a histogram of the sampled coordinate and the reference probability on
the same bins, it returns the next table or refuses. Two things have to be shown about
machinery like that, and neither can be shown by running it on the real force field:

  * that it converges to the right answer. Here the reference is a synthetic double well with
    a known piecewise-linear potential, and the simulator is an EXACT sampler of a coupled
    two-degree-of-freedom model whose stationary marginal is analytic. There is no force
    field and no MCMC error in the loop: every sample is an independent exact draw, so the
    only things under test are the update algebra and the coupling correction. The fixed
    point of the update is known in closed form for both models, so the recovered potential
    can be compared with it bin by bin and the error reported as a number.
  * that it refuses instead of returning something that looks usable. The refusals are
    exercised directly: an unvisited slice of the reference, simulated mass where the
    reference is zero, support drift, and a run held deliberately on the divergent side of
    the stability threshold.

What the synthetic proof does NOT cover: everything about the real force field. The exact
sampler here has no timestep, no thermostat, no truncation, and no unresolved fast degrees of
freedom; the coupling is a mean-field tilt rather than a many-body interaction. A converged
IBI table from this machinery is only as good as the simulation that produced the histogram,
and this file cannot test that.

Run: python tests/test_ibi_bonded.py   (or python -m pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B          # noqa: E402
import ibi_bonded as I                # noqa: E402

NBINS, LO, HI = 90, -1.8, 1.8
EDGES = np.linspace(LO, HI, NBINS + 1)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
BINW = (HI - LO) / NBINS

# The reference potential is a skewed double well, 4*(x^2 - 0.36)^2 + 1.2x, with its range
# clipped to SPAN_KBT so that the tails are flat plateaus. The clip is not cosmetic: a
# quartic tail would put the outermost bins many kBT above the mode, the sampler would never
# visit them, and the update would (correctly) refuse the round -- testing the refusal instead
# of the convergence. Flat tails keep every reference bin measurable while leaving the shape
# clearly non-Gaussian and non-symmetric.
SPAN_KBT = 6.0


def table_of(U, **extra):
    t = {"lo": LO, "hi": HI, "binw": BINW,
         "U": np.asarray(U, dtype=float), "centre": CENTRES}
    t.update(extra)
    return t


def reference_table():
    x = CENTRES
    raw = 4.0 * (x ** 2 - 0.36) ** 2 + 1.2 * x
    return table_of(np.minimum(raw - raw.min(), SPAN_KBT * I.KBT))


def wrong_table():
    """The deliberately wrong starting potential: shallower wells, tilted the other way.

    It has to be wrong in shape without being wrong in support. A potential that leaves a
    region of the reference unsampled does not test convergence at all: the update refuses,
    correctly, and the test would be measuring the refusal.
    """
    x = CENTRES
    U = 2.5 * (x ** 2 - 0.36) ** 2 - 1.6 * x
    return table_of(U - U.min())


# ------------------------------------------------------------------ synthetic engines
def _segment_weights(segs, kBT=I.KBT):
    w = np.array([I.segment_moments(Ua, Ub, q0, q1 - q0, 0.0, 1.0, kBT=kBT)[0]
                  for q0, q1, Ua, Ub in segs])
    return w / w.sum()


def _draw(segs, weights, n, rng):
    """Exact independent draws from exp(-U/kBT) on a tiling segment list.

    Pick a segment with probability proportional to its exact integral, then invert the
    truncated-exponential CDF inside it. No rejection, no MCMC, no discretisation: the only
    approximation is floating point. The draws are filled segment by segment from the segment
    populations, so the cost is O(n) rather than O(n * segments); the order is irrelevant to
    every use here.
    """
    idx = rng.choice(len(segs), size=n, p=weights)
    populations = np.bincount(idx, minlength=len(segs))
    out = np.empty(n, dtype=float)
    pos = 0
    for k, count in enumerate(populations):
        if count == 0:
            continue
        q0, q1, Ua, Ub = segs[k]
        x = (Ub - Ua) / I.KBT
        u = rng.random(int(count))
        if abs(x) < 1e-12:
            t = u
        else:
            t = -np.log1p(-u * (1.0 - np.exp(-x))) / x
        out[pos:pos + count] = q0 + (q1 - q0) * t
        pos += count
    return out


def make_uncoupled_sampler(n, seed=20260911):
    """Engine A: the simulation IS the table's own Boltzmann distribution."""
    state = {"round": 0}

    def sample(table):
        segs = I.sample_segments(table)
        rng = np.random.default_rng(seed + 1000 * state["round"])
        state["round"] += 1
        return _draw(segs, _segment_weights(segs), n, rng)
    return sample


def mean_field_segments(table, J, h):
    """Segments of the effective potential U(q) - J*h*q of a mean-field coupled replica.

    The model is N replicas of the coordinate with a pairwise coupling -(J/2N) sum q_r q_s,
    which in the mean-field limit gives each replica the effective potential above with h the
    self-consistent average. It is not a force field; it is a controllable stand-in for
    "the marginal of the simulation is not exp(-U/kBT)", which is the thing DBI cannot fix.
    """
    return [(q0, q1, Ua - J * h * q0, Ub - J * h * q1)
            for q0, q1, Ua, Ub in I.sample_segments(table)]


def self_consistent_h(table, J, bracket=None, tol=1e-12, iters=200):
    """Solve h = <q> under exp(-(U - J h q)/kBT) by bisection, and verify the root.

    Bisection needs g(h) = <q>(h) - h to change sign; it does, because <q>(h) stays inside
    [q_min, q_max] so g(q_min) >= 0 >= g(q_max). The returned h is checked against the
    definition rather than trusted, so a caller cannot iterate on an un-equilibrated engine.
    """
    lo, hi = (LO, HI) if bracket is None else bracket

    def g(h):
        return I.segments_mean(mean_field_segments(table, J, h)) - h

    g_lo, g_hi = g(lo), g(hi)
    if not (g_lo >= 0.0 >= g_hi):
        raise ValueError(f"no sign change on [{lo}, {hi}]: g = {g_lo}, {g_hi}")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if g(mid) >= 0.0:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi)
    residual = abs(g(h))
    if residual > tol:
        raise ValueError(f"self-consistency not reached: |<q> - h| = {residual}")
    return h, residual


def make_coupled_sampler(n, J, seed=20260912):
    """Engine B: exact sampling of the coupled model, re-equilibrated for every table."""
    state = {"round": 0}

    def sample(table):
        h, _ = self_consistent_h(table, J)
        segs = mean_field_segments(table, J, h)
        rng = np.random.default_rng(seed + 1000 * state["round"])
        state["round"] += 1
        return _draw(segs, _segment_weights(segs), n, rng)
    return sample


def run_ibi(table, p_ref, sample_fn, rounds=8, **policy):
    """The loop a driver runs: sample the current table, update, carry the history."""
    results, tables, history = [], [table], []
    for _ in range(rounds):
        values = sample_fn(tables[-1])
        res = I.advance_from_samples(tables[-1], values, p_ref, history=tuple(history),
                                     **policy)
        results.append(res)
        history.append(res.max_abs_dU)
        if not res.ok:
            break
        tables.append(res.table)
        if res.converged:
            break
    return results, tables


def shape_error(U, U_ref):
    """max |(U - U_ref) - offset| in kJ/mol: the constant is not part of the answer."""
    d = np.asarray(U, dtype=float) - np.asarray(U_ref, dtype=float)
    return float(np.abs(d - d.mean()).max())


def shape_rms(U, U_ref):
    """The same after removing the offset, as an RMS: the typical bin rather than the worst."""
    d = np.asarray(U, dtype=float) - np.asarray(U_ref, dtype=float)
    return float(np.sqrt(((d - d.mean()) ** 2).mean()))


# ------------------------------------------------------------------------ cached runs
N_SYNTHETIC = 8_000_000
# A tolerance below the sampling noise of the histogram is not a convergence criterion, it is
# a coin flip: the iteration would be chasing its own noise. TOL_KBT is set above the measured
# two-histogram floor at N_SYNTHETIC // 4 (see test_a_tolerance_below_the_noise_floor_...),
# which is the stricter of the two sample sizes used here.
TOL_KBT = 0.4
J_COUPLED = 2.2
_CACHE = {}


def cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def uncoupled_run():
    def build():
        ref, wrong = reference_table(), wrong_table()
        p_ref = I.bin_probabilities_from_U(ref)
        results, tables = run_ibi(wrong, p_ref, make_uncoupled_sampler(N_SYNTHETIC), rounds=8,
                                  tol_kbt=TOL_KBT)
        return ref, p_ref, results, tables
    return cached("uncoupled", build)


def coupled_run():
    def build():
        ref, wrong = reference_table(), wrong_table()
        p_ref = I.bin_probabilities_from_U(ref)
        results, tables = run_ibi(wrong, p_ref, make_coupled_sampler(N_SYNTHETIC, J_COUPLED),
                                  rounds=12, tol_kbt=TOL_KBT)
        h_star = I.segments_mean(I.sample_segments(ref))
        return ref, p_ref, results, tables, h_star
    return cached("coupled", build)


def noise_floor(n, seed=4242):
    """Two independent histograms of the SAME table: the floor the update cannot see below."""
    ref = reference_table()
    sampler = make_uncoupled_sampler(n, seed=seed)
    a = I.bin_samples(sampler(ref), ref)
    b = I.bin_samples(sampler(ref), ref)
    pa = I.probability_from_counts(a.counts)
    pb = I.probability_from_counts(b.counts)
    d = np.abs(I.KBT * np.log(pa / pb))
    return float(d.max()), int(d.argmax()), int(a.counts[d.argmax()])


# ------------------------------------------------------------------- the machinery itself
def test_the_exact_fixed_point_is_the_reference_potential():
    """With no sampling noise at all, the update's fixed point is the reference potential.

    The iteration is run deterministically: the 'simulation' is handed the exact bin
    probabilities of the current table instead of a sampled histogram, so the only thing under
    test is the algebra of U <- U + kBT ln(P_sim/P_ref). If the sign, the kBT, the
    normalisation or the bin convention were wrong, this would not land on U_ref, and it lands
    on it to machine precision rather than to a statistical tolerance.
    """
    ref = reference_table()
    p_ref = I.bin_probabilities_from_U(ref)
    table = wrong_table()
    history = []
    for _ in range(300):
        counts = np.round(I.bin_probabilities_from_U(table) * 1e12).astype(np.int64)
        res = I.plan_update(table, I.SimHistogram(
            counts=counts, n=int(counts.sum()), n_outside=0, lo=LO, hi=HI, nbins=NBINS),
            p_ref, history=tuple(history), min_bin_obs=1)
        assert res.ok, (res.status, res.reason)
        table = res.table
        history.append(res.max_abs_dU)
    error = shape_error(table["U"], ref["U"])
    assert error < 1e-6, error


def test_synthetic_recovery_from_a_deliberately_wrong_start():
    """Acceptance 1, uncoupled: exact samples of each trial table, start far from the answer.

    The simulator is exact for the table it is given, so the coupling correction is identically
    zero and the fixed point is the reference potential itself. This is the sign-and-algebra
    test with a distribution in it: a skewed double well with flat tails, not a Gaussian.
    """
    ref, _, results, tables = uncoupled_run()
    assert results[-1].converged, [(r.status, r.reason) for r in results]
    assert results[0].max_abs_dU > 2.0 * I.KBT          # the start really is wrong
    recovered = tables[-1]["U"]
    error = shape_error(recovered, ref["U"])
    assert error < 0.25 * I.KBT, error
    # the typical bin is far better than the worst one, which sits in the emptiest bin
    assert shape_rms(recovered, ref["U"]) < 0.05 * I.KBT, shape_rms(recovered, ref["U"])
    # the well is located to within one bin of the reference minimum
    assert abs(int(np.argmin(recovered)) - int(np.argmin(ref["U"]))) <= 1


def test_coupled_iteration_reaches_the_analytic_fixed_point():
    """Acceptance 1, coupled: the marginal of the simulation is not exp(-U/kBT).

    The model is a mean-field coupled replica, so the simulation's q-marginal is the Boltzmann
    distribution of U(q) - J*h*q with h the self-consistent average, and the update has to
    compensate for that tilt. The fixed point is known in closed form: the table must become
    U_ref(q) + J*h_star*q with h_star = <q> of the reference, independent of J. The run has to
    take more than one round, or the coupling was not being corrected at all.
    """
    ref, _, results, tables, h_star = coupled_run()
    assert results[-1].converged, [(r.status, r.reason) for r in results]
    assert len(results) > 1, "the coupled case converged in one round, so it is not coupled"
    u_star = ref["U"] + J_COUPLED * h_star * CENTRES
    error = shape_error(tables[-1]["U"], u_star)
    assert error < 0.25 * I.KBT, error
    # and the coupling is not a decoration: ignoring it lands measurably further away
    assert shape_error(tables[-1]["U"], ref["U"]) > 2.0 * error


def test_a_divergent_iteration_is_refused_and_returns_no_table():
    """Acceptance 2: an over-relaxed update (gain 2.5, past the stability limit) runs away.

    Measured on this reference: the applied correction rises 20.6, 27.6, 28.1, 48.7 kJ/mol over
    four rounds and the growth rule stops it at the fourth, before the absolute ceiling is
    reached. The point of the test is that the run ends with no table at all.
    """
    ref, wrong = reference_table(), wrong_table()
    p_ref = I.bin_probabilities_from_U(ref)
    results, _ = run_ibi(wrong, p_ref, make_uncoupled_sampler(1_000_000), rounds=8, gain=2.5)
    last = results[-1]
    assert not last.ok and last.table is None
    assert last.status == I.STATUS_REFUSED
    assert last.reason in (I.REFUSE_DIVERGENCE, I.REFUSE_STEP_TOO_LARGE), last.reason
    with pytest.raises(I.IBIRefusal):
        last.require_table()
    # the corrections really did grow: this is divergence, not one unlucky round
    steps = [r.max_abs_dU for r in results]
    assert all(b >= a for a, b in zip(steps, steps[1:])), steps
    assert steps[-1] > 2.0 * steps[0], steps


def test_a_single_round_above_the_step_ceiling_is_refused():
    """The absolute ceiling, on its own: one round trying to move 8 kJ/mol with a 2 kBT cap."""
    ref, wrong = reference_table(), wrong_table()
    p_ref = I.bin_probabilities_from_U(ref)
    res = I.advance_from_samples(wrong, make_uncoupled_sampler(200_000)(wrong), p_ref,
                                 max_step_kbt=2.0)
    assert res.reason == I.REFUSE_STEP_TOO_LARGE, res.reason
    assert res.table is None
    assert res.diagnostics["max_abs_dU"] > 2.0 * I.KBT


def test_the_growth_rule_fires_on_a_rising_correction():
    """The same run with the ceiling lifted: the growth rule is what stops it."""
    ref, wrong = reference_table(), wrong_table()
    p_ref = I.bin_probabilities_from_U(ref)
    results, _ = run_ibi(wrong, p_ref, make_uncoupled_sampler(1_000_000), rounds=8, gain=2.5,
                         max_step_kbt=200.0)
    last = results[-1]
    assert last.reason == I.REFUSE_DIVERGENCE, last.reason
    assert last.table is None
    history = [r.max_abs_dU for r in results[:-1]]
    assert last.max_abs_dU >= 2.0 * history[0]


def test_the_divergence_rule_ignores_a_settling_correction():
    """A flat or falling correction must never be called divergent, or the rule is useless."""
    assert I.divergence_check([], 0.5) == (False, None)
    assert I.divergence_check([1.0, 0.6, 0.3], 0.1)[0] is False
    assert I.divergence_check([0.1, 0.1, 0.1], 0.1)[0] is False
    # rising but without the growth factor: not divergent yet
    assert I.divergence_check([1.0, 1.1, 1.2], 1.3)[0] is False
    diverging, message = I.divergence_check([1.0, 2.0, 4.0], 8.0)
    assert diverging and "rose for 3" in message
    assert I.divergence_check([1.0], 5.0, ceiling=4.0)[0] is True
    assert I.divergence_check([1.0, 2.0, 4.0], 8.0, patience=4)[0] is False


# --------------------------------------------------------------- refusals and bookkeeping
def hist_from(p, n, n_outside=0):
    counts = np.floor(np.asarray(p, dtype=float) * n).astype(np.int64)
    total = int(counts.sum()) + n_outside
    return I.SimHistogram(counts=counts, n=total, n_outside=n_outside, lo=LO, hi=HI,
                          nbins=NBINS)


def test_an_unvisited_slice_of_the_reference_is_refused():
    ref = reference_table()
    p_ref = I.bin_probabilities_from_U(ref)
    counts = np.floor(p_ref * 1_000_000).astype(np.int64)
    missing = counts > 0
    order = np.argsort(-p_ref)
    order = order[:40]                      # the 40 most populated bins (~40 percent of mass)
    counts[order] = 0
    hist = I.SimHistogram(counts=counts, n=int(counts.sum()), n_outside=0, lo=LO, hi=HI,
                          nbins=NBINS)
    res = I.plan_update(ref, hist, p_ref)
    assert res.table is None and res.status == I.STATUS_REFUSED
    assert res.reason == I.REFUSE_UNSUPPORTED, res.reason
    assert res.diagnostics["unvisited_ref_mass"] > I.DEFAULT_MAX_UNSUPPORTED_MASS
    assert res.diagnostics["message"]


def test_a_small_unmeasured_gap_is_frozen_and_blocks_convergence():
    """Below the mass threshold a round is usable, but the gap is carried, not interpolated."""
    ref = reference_table()
    p_ref = I.bin_probabilities_from_U(ref)
    counts = np.floor(p_ref * 1_000_000).astype(np.int64)
    gap = int(np.argmin(p_ref))
    counts[gap] = 0
    hist = I.SimHistogram(counts=counts, n=int(counts.sum()), n_outside=0, lo=LO, hi=HI,
                          nbins=NBINS)
    res = I.plan_update(ref, hist, p_ref)
    assert res.ok and res.status == I.STATUS_OK
    assert not res.converged, "a round with an unmeasured bin cannot be called converged"
    assert res.diagnostics["unvisited_bins"] == 1
    assert res.diagnostics["unvisited_ref_mass"] == pytest.approx(float(p_ref[gap]))
    # frozen, not fabricated: that bin keeps its previous value up to the global shift
    assert res.table["U"][gap] == pytest.approx(ref["U"][gap] - res.diagnostics["shift"])
    # and it is not equal to what the pseudocount would have produced there
    fabricated = I.KBT * np.log(I.probability_from_counts(counts)[gap] / p_ref[gap])
    assert abs(res.table["U"][gap] - (ref["U"][gap] + fabricated)) > 0.5 * I.KBT


def test_simulated_mass_where_the_reference_is_zero_is_refused():
    ref = reference_table()
    p_ref = I.bin_probabilities_from_U(ref)
    p_ref = p_ref.copy()
    p_ref[0] = 0.0
    counts = np.floor(p_ref * 1_000_000).astype(np.int64)
    counts[0] = 25
    hist = I.SimHistogram(counts=counts, n=int(counts.sum()), n_outside=0, lo=LO, hi=HI,
                          nbins=NBINS)
    res = I.plan_update(ref, hist, p_ref)
    assert res.table is None and res.reason == I.REFUSE_SIM_WHERE_REF_ZERO, res.reason
    assert res.diagnostics["sim_count_where_ref_zero"] == 25


def test_support_drift_is_refused():
    ref = reference_table()
    p_ref = I.bin_probabilities_from_U(ref)
    res = I.plan_update(ref, hist_from(p_ref, 1_000_000, n_outside=50_000), p_ref)
    assert res.table is None and res.reason == I.REFUSE_SUPPORT_DRIFT, res.reason
    assert res.diagnostics["outside_frac"] == pytest.approx(0.05, rel=0.05)
    # the same histogram under the support is accepted, so the refusal is about the drift
    assert I.plan_update(ref, hist_from(p_ref, 1_000_000, n_outside=1), p_ref).ok


def test_no_samples_at_all_is_refused():
    ref = reference_table()
    p_ref = I.bin_probabilities_from_U(ref)
    hist = I.SimHistogram(counts=np.zeros(NBINS, dtype=np.int64), n=100, n_outside=100,
                          lo=LO, hi=HI, nbins=NBINS)
    res = I.plan_update(ref, hist, p_ref)
    assert res.table is None and res.reason == I.REFUSE_NO_SAMPLES, res.reason
    with pytest.raises(I.IBIRefusal):
        res.require_table()


def test_thin_bins_are_reported_and_do_not_refuse_by_themselves():
    ref = reference_table()
    p_ref = I.bin_probabilities_from_U(ref)
    counts = np.floor(p_ref * 1_000_000).astype(np.int64)
    thin = int(np.argmin(p_ref))
    counts[thin] = 3
    hist = I.SimHistogram(counts=counts, n=int(counts.sum()), n_outside=0, lo=LO, hi=HI,
                          nbins=NBINS)
    res = I.plan_update(ref, hist, p_ref)
    assert res.ok
    assert res.diagnostics["thin_bins"] == 1
    assert res.diagnostics["thin_counts"] == 3
    assert res.diagnostics["min_bin_obs"] == I.DEFAULT_MIN_BIN_OBS


def test_a_histogram_from_different_bins_is_rejected():
    ref = reference_table()
    p_ref = I.bin_probabilities_from_U(ref)
    good = hist_from(p_ref, 100_000)
    I.plan_update(ref, good, p_ref)                    # the honest histogram is accepted
    for bad in (good._replace(nbins=NBINS + 1), good._replace(lo=LO - 0.1),
                good._replace(hi=HI + 0.1), good._replace(n=good.n + 1),
                good._replace(n_outside=1)):
        with pytest.raises((ValueError, TypeError)):
            I.plan_update(ref, bad, p_ref)
    with pytest.raises(ValueError):
        I.plan_update(ref, good, p_ref[:NBINS - 1])
    with pytest.raises(ValueError):
        I.plan_update(ref, good, np.zeros(NBINS))


# ------------------------------------------------------------- format and conventions
def test_the_returned_table_has_exactly_the_keys_prepare_reads():
    ref, _, results, tables = uncoupled_run()
    table = tables[-1]
    assert set(table) == set(I.TABLE_KEYS)
    assert "sigma" not in table, ("prepare() would inject sigma=0.0 only if it were absent, "
                                  "and mixed_energy then defaults k_local to kBT/sigma^2")
    prepared = B.prepare({"angle": table})
    pos = torch.randn(1, 3 * 6, 3, dtype=torch.float64) * 0.3
    energy = B.mixed_energy(pos, prepared, which=("angle",))
    assert torch.isfinite(energy) and energy.ndim == 0
    assert table["n"] == results[-1].diagnostics["n_sim"]
    assert table["empty"] == results[-1].diagnostics["unvisited_bins"]


def test_bin_samples_counts_what_np_histogram_counts_and_reports_the_rest():
    ref = reference_table()
    values = np.concatenate([np.linspace(LO, HI, 10_000),
                             np.array([LO - 0.01, HI + 0.01, LO - 5.0])])
    hist = I.bin_samples(values, ref)
    direct, _ = np.histogram(values, bins=NBINS, range=(LO, HI))
    assert np.array_equal(hist.counts, direct)
    assert hist.n == values.size
    assert hist.n_outside == 3
    assert (hist.lo, hist.hi, hist.nbins) == (LO, HI, NBINS)
    assert hist.counts.sum() == hist.n - hist.n_outside


def test_the_pseudocount_policy_is_boltzmann_bondeds():
    rng = np.random.default_rng(3)
    v = np.concatenate([rng.normal(0.0, 1.0, 5000), rng.normal(3.0, 0.5, 500)])
    shipped = B._table_from_values(v, nbins=40, pseudo=0.5)
    counts, _ = np.histogram(v, bins=40, range=(shipped["lo"], shipped["hi"]))
    mine = -I.KBT * np.log(I.probability_from_counts(counts, 0.5))
    assert np.allclose(mine - mine.min(), shipped["U"], atol=1e-12)


def test_reference_probability_is_the_ibi_round0_convention():
    """ibi_round0.ref_p: p = exp(-(U - U.min())/kBT), normalised. Same numbers or the
    reference the loop corrects towards is not the one that was measured."""
    t = I.load_clean_tables(REPO / "results" / "boltzmann_tables_clean.npz")["angle"]
    U = t["U"]
    inline = np.exp(-(U - U.min()) / B.KBT)
    inline = inline / inline.sum()
    assert np.allclose(I.probability_from_U(U), inline, rtol=1e-15, atol=0)
    assert np.allclose(I.probability_from_table(t), inline, rtol=1e-15, atol=0)


def test_smoothing_is_a_normalised_moving_average_and_identity_at_zero():
    dU = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(I.smooth_correction(dU, 0), dU)
    flat = I.smooth_correction(np.full(7, 2.5), 2)
    assert np.allclose(flat, 2.5, rtol=0, atol=1e-15)
    spike = I.smooth_correction(np.array([0.0, 0.0, 9.0, 0.0, 0.0]), 1)
    assert np.allclose(spike[1:4], [3.0, 3.0, 3.0], atol=1e-15)
    # a masked bin contributes to nobody and is left alone, so a gap cannot leak
    masked = I.smooth_correction(np.array([0.0, 100.0, 0.0]), 1,
                                 mask=np.array([True, False, True]))
    assert np.allclose(masked, [0.0, 100.0, 0.0], atol=1e-15)
    with pytest.raises(ValueError):
        I.smooth_correction(dU, 3)


def test_smoothing_shrinks_the_correction_and_says_so():
    ref = reference_table()
    p_ref = I.bin_probabilities_from_U(ref)
    hist = hist_from(p_ref, 1_000_000)
    plain = I.plan_update(ref, hist, p_ref)
    smoothed = I.plan_update(ref, hist, p_ref, smooth_bins=2)
    assert smoothed.diagnostics["smoothed"] and smoothed.diagnostics["smooth_bins"] == 2
    assert not plain.diagnostics["smoothed"]
    assert smoothed.max_abs_dU <= plain.max_abs_dU


# -------------------------------------------------------- the two reference conventions
def deterministic_fixed_point(p_target, table, rounds=300):
    """The update with the sampling replaced by the exact bin probabilities."""
    current = table
    for _ in range(rounds):
        p_sim = I.bin_probabilities_from_U(current)
        counts = np.round(p_sim * 1e12)
        res = I.plan_update(current, I.SimHistogram(
            counts=counts, n=int(counts.sum()), n_outside=0, lo=LO, hi=HI, nbins=NBINS),
            p_target, min_bin_obs=1)
        assert res.ok, (res.status, res.reason)
        current = res.table
    return current


def test_the_two_reference_conventions_do_not_share_a_fixed_point():
    """The bin integral and the point weight of the same table are different objects.

    boltzmann_bonded stores U = -kBT ln p_bin, so exp(-U/kBT) is the bin probability and the
    bin integral of the *same* table is not: the two differ by kBT ln[(1 - e^-x)/x] with x the
    step across a bin, about kBT*x/2. A simulator reads the table as a function, so it
    reproduces the bin integral, and the update moves the table by the difference. Measured
    here on the synthetic reference, and pushed to its fixed point with no sampling noise.
    """
    ref = reference_table()
    p_bins = I.bin_probabilities_from_U(ref)
    p_point = I.probability_from_U(ref["U"])
    predicted = I.KBT * np.log(p_bins / p_point)
    predicted = predicted - predicted.mean()
    assert np.abs(predicted).max() > 0.05 * I.KBT      # the two conventions really differ

    to_bins = deterministic_fixed_point(p_bins, wrong_table())
    assert shape_error(to_bins["U"], ref["U"]) < 1e-6  # this one is the reference potential

    to_point = deterministic_fixed_point(p_point, wrong_table())
    shift = to_point["U"] - ref["U"]
    shift = shift - shift.mean()
    assert shape_error(to_point["U"], ref["U"]) > 0.05 * I.KBT
    assert np.abs(shift - predicted).max() < 0.05 * I.KBT


def test_the_analytic_bin_probabilities_match_the_shipped_sample():
    """bin_probabilities_from_U has to be the integral of the function the pipeline samples,
    flat clamped ends included, or the reference it builds is a different distribution from
    the one the force field reproduces."""
    ref = reference_table()
    analytic = I.bin_probabilities_from_U(ref)
    q = np.linspace(LO, HI, 200_001)
    sampled_u = B._sample(torch.tensor(q), {"lo": LO, "hi": HI, "binw": BINW,
                                            "U": torch.tensor(ref["U"]),
                                            "Ut": torch.tensor(ref["U"])}).numpy()
    density = np.exp(-(sampled_u - sampled_u.min()) / I.KBT)
    grid = np.zeros(NBINS)
    index = np.clip(((q - LO) / BINW).astype(int), 0, NBINS - 1)
    for j in range(NBINS):
        m = index == j
        grid[j] = np.trapezoid(density[m], q[m])
    grid = grid / grid.sum()
    assert np.abs(analytic - grid).max() < 5e-5
    # the flat ends are real mass, not a rounding artifact: dropping them changes the answer
    assert analytic[0] > 0 and analytic[-1] > 0


# ------------------------------------------------------------------- the real artefacts
def test_load_clean_tables_reads_what_ibi_round0_reads():
    tables = I.load_clean_tables(REPO / "results" / "boltzmann_tables_clean.npz")
    assert set(tables) == set(B.COORDS)
    for name, t in tables.items():
        assert set(t) == {"lo", "hi", "binw", "U", "centre", "sigma"}, name
        assert t["U"].shape == (120,)
        assert t["binw"] == pytest.approx((t["hi"] - t["lo"]) / 120, rel=1e-12)
    B.prepare({k: dict(v) for k, v in tables.items()})


def test_a_stored_table_drives_one_real_round():
    """The handover ibi_round0 needs: stored table -> exact sampler -> one update round.

    The correction that comes back is the bin-integral convention term, not physics: the
    simulator is given the stored table itself, so there is no coupling and no missing term.
    It has to come out close to kBT ln(P_bins/P_pointweight) computed directly from the same
    table, which is the size of the first-round move a driver should expect to see before any
    coupling enters.
    """
    table = I.load_clean_tables(REPO / "results" / "boltzmann_tables_clean.npz")["stack"]
    p_ref = I.probability_from_table(table)
    values = make_uncoupled_sampler(2_000_000, seed=99)(table)
    res = I.advance_from_samples(table, values, p_ref, min_bin_obs=I.DEFAULT_MIN_BIN_OBS)
    assert res.ok, (res.status, res.reason)
    assert set(res.table) == set(I.TABLE_KEYS)
    predicted = I.KBT * np.log(I.bin_probabilities_from_U(table) / p_ref)
    predicted = predicted - predicted.mean()
    measured = res.dU - res.dU.mean()
    assert np.abs(measured - predicted).max() < 0.2 * I.KBT
    # and the round moved the table towards the target it was given
    before = np.abs(I.bin_probabilities_from_U(table) - p_ref).max()
    after = np.abs(I.bin_probabilities_from_U(res.table) - p_ref).max()
    assert after < before


def test_a_tolerance_below_the_noise_floor_cannot_represent_convergence():
    """Why TOL_KBT is 0.4: two histograms of the same table differ by this much."""
    floor, _, counts = noise_floor(N_SYNTHETIC // 4)
    assert TOL_KBT * I.KBT > floor, (TOL_KBT * I.KBT, floor)
    assert 2 * TOL_KBT * I.KBT > floor, "the margin is too thin to be a margin"


def report():
    """The measured numbers behind this file's claims. Run the file directly to see them."""
    print()
    print("=" * 88)
    print("measured numbers (synthetic reference: skewed double well, flat tails, "
          f"{NBINS} bins on [{LO}, {HI}])")
    print("=" * 88)
    ref, p_ref, results, tables = uncoupled_run()
    print(f"uncoupled engine, {N_SYNTHETIC} exact samples per round, tol {TOL_KBT} kBT")
    for k, r in enumerate(results):
        print(f"   round {k}: {r.status:9s} max|dU| {r.max_abs_dU:8.4f} kJ/mol "
              f"({r.max_abs_dU / I.KBT:6.4f} kBT)  unvisited {r.diagnostics['unvisited_bins']}"
              f"  thin {r.diagnostics['thin_bins']}")
    e = shape_error(tables[-1]["U"], ref["U"])
    r = shape_rms(tables[-1]["U"], ref["U"])
    print(f"   recovered vs the reference potential: max|dU| {e:.4f} kJ/mol = {e / I.KBT:.4f} kBT"
          f", rms {r:.4f} kJ/mol = {r / I.KBT:.4f} kBT")
    print(f"   rounds from the deliberately wrong start: {len(results)} "
          f"(round 0 correction was {results[0].max_abs_dU:.3f} kJ/mol)")

    ref, _, results, tables, h_star = coupled_run()
    print()
    print(f"coupled engine (mean-field replica, J = {J_COUPLED} kJ/mol/nm^2), "
          f"{N_SYNTHETIC} exact samples per round")
    for k, r in enumerate(results):
        print(f"   round {k}: {r.status:9s} max|dU| {r.max_abs_dU:8.4f} kJ/mol "
              f"({r.max_abs_dU / I.KBT:6.4f} kBT)  unvisited {r.diagnostics['unvisited_bins']}")
    u_star = ref["U"] + J_COUPLED * h_star * CENTRES
    e = shape_error(tables[-1]["U"], u_star)
    print(f"   h_star = <q> of the reference = {h_star:.6f} nm; the tilt spans "
          f"{J_COUPLED * h_star * (HI - LO):.3f} kJ/mol")
    r = shape_rms(tables[-1]["U"], u_star)
    print(f"   recovered vs the analytic coupled fixed point U_ref + J*h_star*q: "
          f"{e:.4f} kJ/mol = {e / I.KBT:.4f} kBT, rms {r:.4f} kJ/mol = {r / I.KBT:.4f} kBT")
    print(f"   (vs U_ref alone, i.e. ignoring the coupling: "
          f"{shape_error(tables[-1]['U'], ref['U']):.4f} kJ/mol)")

    floor, worst, counts = noise_floor(N_SYNTHETIC // 4)
    print()
    print(f"noise floor: two independent histograms of the same table, "
          f"{N_SYNTHETIC // 4} samples each -> max|dU| {floor:.4f} kJ/mol "
          f"= {floor / I.KBT:.4f} kBT (bin {worst}, {counts} counts)")
    print(f"   TOL_KBT = {TOL_KBT} kBT = {TOL_KBT * I.KBT:.4f} kJ/mol, above that floor by "
          f"a factor {(TOL_KBT * I.KBT) / floor:.2f}")

    print()
    print("divergence, over-relaxed update (gain 2.5), default ceiling "
          f"{I.DEFAULT_MAX_STEP_KBT} kBT:")
    res, _ = run_ibi(wrong_table(), p_ref, make_uncoupled_sampler(1_000_000), rounds=8, gain=2.5)
    steps = ", ".join(f"{r.max_abs_dU:.2f}" for r in res)
    print(f"   {len(res)} rounds, max|dU| {steps} kJ/mol")
    print(f"   refused with reason {res[-1].reason!r}: "
          f"{res[-1].diagnostics['message']}")
    res = I.advance_from_samples(wrong_table(), make_uncoupled_sampler(200_000)(wrong_table()),
                                 p_ref, max_step_kbt=2.0)
    print(f"   one round with a 2 kBT ceiling: reason {res.reason!r}, "
          f"max|dU| {res.max_abs_dU:.3f} kJ/mol, table {res.table}")

    p_bins = I.bin_probabilities_from_U(ref)
    p_point = I.probability_from_U(ref["U"])
    predicted = I.KBT * np.log(p_bins / p_point)
    predicted = predicted - predicted.mean()
    to_bins = deterministic_fixed_point(p_bins, wrong_table())
    to_point = deterministic_fixed_point(p_point, wrong_table())
    print()
    print("reference conventions, exact iteration (no sampling):")
    print(f"   bin-integral target fixed point vs the reference potential: "
          f"{shape_error(to_bins['U'], ref['U']):.3e} kJ/mol")
    print(f"   point-weight target fixed point vs the reference potential: "
          f"{shape_error(to_point['U'], ref['U']):.4f} kJ/mol; predicted shape "
          f"max|kBT ln(P_bins/P_point)| = {np.abs(predicted).max():.4f} kJ/mol")

    print()
    print("stored tables (results/boltzmann_tables_clean.npz): the same convention gap, "
          "computed from the file")
    stored = I.load_clean_tables(REPO / "results" / "boltzmann_tables_clean.npz")
    for name in ("angle", "dihedral", "stack"):
        t = stored[name]
        shift = I.KBT * np.log(I.bin_probabilities_from_U(t) / I.probability_from_U(t["U"]))
        shift = shift - shift.mean()
        p = I.probability_from_U(t["U"])
        order = np.sort(p)[::-1]
        core = p >= order[min(int((np.cumsum(order) < 0.99).sum()), len(order) - 1)]
        print(f"   {name:9s} U spans {t['U'].max() - t['U'].min():6.2f} kJ/mol; "
              f"convention gap max {np.abs(shift).max():6.3f} kJ/mol over all bins, "
              f"{np.abs(shift[core]).max():6.3f} kJ/mol = "
              f"{np.abs(shift[core]).max() / I.KBT:.3f} kBT over the 99 percent mass core")

    table = stored["stack"]
    p_table = I.probability_from_table(table)
    res = I.advance_from_samples(table, make_uncoupled_sampler(2_000_000, seed=99)(table),
                                 p_table)
    meas = res.dU - res.dU.mean()
    pred = I.KBT * np.log(I.bin_probabilities_from_U(table) / p_table)
    pred = pred - pred.mean()
    print(f"   one real round on the stored stack table: status {res.status}, "
          f"max|dU| {res.max_abs_dU:.4f} kJ/mol; measured vs predicted convention shift "
          f"max {np.abs(meas - pred).max():.4f} kJ/mol = "
          f"{np.abs(meas - pred).max() / I.KBT:.4f} kBT")


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
    report()
    sys.exit(1 if failed else 0)


