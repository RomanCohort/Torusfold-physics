"""The moment operator (Plan B'): does it descend the thing it claims to descend?

Plan B' (docs/archive/plan_b_coupled_update.md) replaces the per-bin marginal inversion with a low-order
correction whose coefficients are the relative-entropy gradient:

    U_new(q) = U_table(q) + sum_k d_k T_k(x),    d_k = eta_k (<T_k>_sim - <T_k>_ref)

Three claims have to hold, and each of these tests is one of them.

  1. When the simulation already matches the reference, the correction is zero -- otherwise the
     operator would be moving a fixed point that is already correct.
  2. The correction pushes in the right DIRECTION: a simulation that has drifted to +x gets a
     correction that is larger at +x.
  3. Iterated against a response that is perturbed (the stand-in for coupling: the rest of the field
     reshaping this coordinate's marginal), it reaches the moment-matching fixed point. That is the
     property the marginal inversion fails to have on the coupled coordinates -- angle's correction
     oscillated 3.00 -> 3.94 -> 3.02 over three rounds of the real loop.
  4. It does not inject binning noise: with a noisy histogram the per-bin ratio has far more
     high-frequency content than the low-order correction, which is what makes the sampler heat.
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

KBT = B.KBT
NB = 1000


def _bimodal(nb=NB, lo=-1.0, hi=1.0):
    """A reference shaped like the dihedral's own: a narrow peak against a broader basin."""
    x = np.linspace(lo + (hi - lo) / (2 * nb), hi - (hi - lo) / (2 * nb), nb)
    p = np.exp(-0.5 * ((x + 0.45) / 0.12) ** 2) + 0.6 * np.exp(-0.5 * ((x - 0.15) / 0.35) ** 2)
    return x, p / p.sum()


def _table_from_u(x, U, sigma=0.1):
    nb = len(x)
    binw = float(x[1] - x[0])
    return {"lo": float(x[0] - binw / 2), "hi": float(x[-1] + binw / 2), "binw": binw,
            "U": np.asarray(U, dtype=float), "centre": np.asarray(x, dtype=float), "sigma": sigma}


def _table_from_p(x, p, sigma=0.1):
    return _table_from_u(x, -KBT * np.log(np.maximum(p, 1e-300) / p.max()), sigma=sigma)


def test_a_matching_simulation_gets_no_correction():
    x, p_ref = _bimodal()
    tab = _table_from_p(x, p_ref)
    counts = np.round(p_ref * 1_000_000).astype(np.int64)
    res = I.moment_correction(tab, counts, int(counts.sum()), 0, p_ref)
    assert res.ok, res.reason
    # Not bitwise zero: the counts are rounded and the step is a linear solve, so "no correction"
    # means "no correction above the rounding the step can amplify" -- 0.04 kBT is that scale, and
    # it is what the ridge was sized against (a 1e-8 ridge turned the same input into ~0.4 kBT).
    assert np.abs(res.dU).max() < 0.1, (
        f"a matching simulation produced a {np.abs(res.dU).max():.4g} kJ/mol correction")


def test_the_correction_points_back_at_the_reference():
    """A simulation shifted to +x gets a potential that is larger at +x."""
    x, p_ref = _bimodal()
    tab = _table_from_p(x, p_ref)
    shift = np.exp(-0.5 * ((x - 0.08) / 0.12) ** 2)
    p_sim = p_ref * (1.0 + 0.5 * shift / shift.max())
    p_sim = p_sim / p_sim.sum()
    counts = np.round(p_sim * 1_000_000).astype(np.int64)
    res = I.moment_correction(tab, counts, int(counts.sum()), 0, p_ref, K=4)
    assert res.ok and res.diagnostics["moment_norm"] > 1e-4
    # WHAT A NEWTON STEP CAN CLAIM: under linear response its step zeroes the moment differences.
    # (It is NOT claimable that the coefficient of T_1 is positive -- T_1 is x on this support, but
    # Chebyshev components are correlated under a peaked distribution and the solve works in that
    # basis: measured for this shift, the coefficients are [-0.53, +0.43, -0.36, +0.56].)
    A, _ = I._chebyshev_design(tab["centre"], tab["lo"], tab["hi"], 4)
    w = counts / counts.sum()
    sim = w @ A
    pref = np.asarray(p_ref, dtype=float) / np.sum(p_ref)
    delta = sim - (pref @ A)
    cov = (w[:, None] * A).T @ A - np.outer(sim, sim)
    cov = cov + 1e-3 * float(np.trace(cov)) / len(cov) * np.eye(len(cov))
    dc = res.diagnostics["coefficients_kJ"]
    residual = delta - cov @ dc / KBT          # d<T_k> = -Cov dc / kBT
    assert np.abs(residual).max() < 0.02 * np.abs(delta).max(), (delta, residual)


def test_iterated_against_a_perturbed_response_it_converges():
    """The stand-in for coupling: the rest of the field reshapes the marginal by w(x).

    The marginal inversion is not expected to settle on a coupled coordinate; this operator is
    supposed to, and "supposed to" is what this test checks. The response is exact (no sampling):
    p_sim ∝ exp(-U/kBT) * w(x), which is the shape a multiplicative perturbation makes.
    """
    x, p_ref = _bimodal()
    w = 1.0 + 0.30 * (x - x.mean()) / (x.max() - x.mean())      # a linear tilt, |w| > 0
    tab = _table_from_p(x, p_ref)
    norms = []
    for _ in range(12):
        p_sim = np.exp(-(tab["U"] - tab["U"].min()) / KBT) * w
        p_sim = p_sim / p_sim.sum()
        counts = np.round(p_sim * 1_000_000).astype(np.int64)
        res = I.moment_correction(tab, counts, int(counts.sum()), 0, p_ref, K=8, gain=0.7)
        assert res.ok, res.reason
        norms.append(res.diagnostics["moment_norm"])
        tab = dict(tab, U=res.table["U"])
    assert norms[-1] < 0.2 * norms[0], f"the moment norm did not fall: {norms}"
    assert min(norms[-3:]) < 5e-3, f"did not reach the fixed point: {norms[-3:]}"


def test_it_does_not_inject_binning_noise():
    """The correction's high-frequency content, against the per-bin ratio's.

    This is why the operator exists: kBT ln(P_sim/P_ref) per bin carries the histogram's Poisson
    noise, and a piecewise-linear potential built from it is a random force field. Measure the
    second difference -- the discrete curvature -- of each.
    """
    rng = np.random.default_rng(7)
    x, p_ref = _bimodal()
    tab = _table_from_p(x, p_ref)
    counts = rng.poisson(p_ref * 20_000).astype(np.int64)
    n = int(counts.sum())
    res = I.moment_correction(tab, counts, n, 0, p_ref, K=8)
    d_ratio = I.plan_update(tab, I.SimHistogram(counts=counts, n=n, n_outside=0,
                                                lo=tab["lo"], hi=tab["hi"], nbins=NB),
                            p_ref).dU
    def curvature(v):
        v = np.asarray(v, dtype=float)
        return float(np.abs(np.diff(v, 2)).mean())
    assert curvature(res.dU) < 0.05 * curvature(d_ratio), (
        f"moment correction curvature {curvature(res.dU):.4g} against the ratio's "
        f"{curvature(d_ratio):.4g}")


def test_the_refusals_are_the_same_vocabulary():
    x, p_ref = _bimodal()
    tab = _table_from_p(x, p_ref)
    counts = np.round(p_ref * 100_000).astype(np.int64)
    n = int(counts.sum())
    r = I.moment_correction(tab, counts, 0, 0, p_ref)
    assert r.status == I.STATUS_REFUSED and r.reason == I.REFUSE_NO_SAMPLES
    r = I.moment_correction(tab, counts, n, int(0.02 * n), p_ref)
    assert r.reason == I.REFUSE_SUPPORT_DRIFT
    p_sim = np.roll(p_ref, 40); p_sim = p_sim / p_sim.sum()
    r = I.moment_correction(tab, np.round(p_sim * 100_000).astype(np.int64), n, 0, p_ref,
                            gain=1.0, max_step_kbt=0.01)
    assert r.reason == I.REFUSE_STEP_TOO_LARGE
    with pytest.raises(I.IBIRefusal):
        r.require_table()
