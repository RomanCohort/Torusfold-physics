"""The basis families must agree with the loop where they overlap, or every comparison is void.

plan_c_basis.fit is the C2s pipeline with a pluggable design; plan_c_loop._chebyshev_fit_stable is
the pipeline the arms actually ran. They have to be the same function on the Chebyshev design -- same
support cut, same weights, same trace/m ridge, same taper, same gauge -- because section 6's whole
argument is "same fit, only the design changed".
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ibi_bonded as I          # noqa: E402
import plan_c_basis as PB       # noqa: E402
import plan_c_loop as L         # noqa: E402


def _table(nbins=200, lo=-1.0, hi=1.0, n=40_000, seed=5):
    rng = np.random.default_rng(seed)
    counts, _ = np.histogram(rng.normal(0.2, 0.25, size=n), bins=np.linspace(lo, hi, nbins + 1))
    centre = 0.5 * (np.linspace(lo, hi, nbins + 1)[:-1] + np.linspace(lo, hi, nbins + 1)[1:])
    U = 8.0 * (centre - 0.2) ** 2
    return {"lo": lo, "hi": hi, "binw": (hi - lo) / nbins, "centre": centre,
            "U": U, "sigma": 0.1}, counts.astype(np.int64)


def test_chebyshev_path_matches_the_loop():
    """One design, two entry points, identical numbers."""
    tab, counts = _table()
    target = L._target_from_counts(counts)
    U_loop, d_loop = L._chebyshev_fit_stable(tab, target, counts, 8)
    A = PB.design_chebyshev(tab["centre"], tab["lo"], tab["hi"], 8)
    U_new, d_new = PB.fit(A, tab, counts, U_target=target, ridge_rel=1e-1, ridge_form="trace")
    assert np.allclose(U_new, U_loop, atol=1e-10), "the shared fit is not the loop's fit"
    assert d_new["ridge_lambda"] == pytest.approx(d_loop["ridge_lambda"], rel=1e-12)
    assert d_new["mass_offset_removed"] == pytest.approx(d_loop["mass_offset_removed"], rel=1e-12)


def test_bspline_design_is_local_and_normalised():
    """B-splines: rows sum to 1 on the support (a partition of unity, up to the open-knot ends) and
    each basis function is nonzero over only a slice of it, which is the locality the project tunes."""
    tab, _ = _table(nbins=400)
    widths = {}
    for m in (8, 16, 64):
        A = PB.design_bspline(tab["centre"], tab["lo"], tab["hi"], m)
        assert A.shape == (400, m)
        interior = (tab["centre"] > tab["lo"] + 0.15) & (tab["centre"] < tab["hi"] - 0.15)
        assert np.allclose(A[interior].sum(axis=1), 1.0, atol=1e-9)
        # A cubic B-spline's support is (degree+1) knot SPANS, and there are m-degree spans, so the
        # widest basis function covers about 4/(m-3) of the support. That is the locality parameter.
        widths[m] = int((A > 1e-12).sum(axis=0).max())
        assert widths[m] <= 4 * A.shape[0] / (m - 3) + 2, f"m={m} spans {widths[m]} bins"
    assert widths[64] < widths[16] / 2, "locality does not improve with m"


def test_eigenvalue_relative_ridge_is_scale_free_and_trace_ridge_is_not():
    """The measured lesson: trace/m falls with m while eig_max does not, so the same ridge_rel means
    different things at different m. On the dihedral's own ensemble (2026-09-24) trace/m/eig_max is
    0.430 at m=8 and 0.049 at m=128 -- a factor of 8.8 -- which is why the Chebyshev-tuned ridge_rel
    stops regularising a fine B-spline. This test uses a wide target so the effect is visible."""
    tab, counts = _table()
    counts = (counts + 30).astype(np.int64)      # a broad, edge-carrying histogram
    target = L._target_from_counts(counts)
    lam = {}
    for m in (8, 64):
        A = PB.design_bspline(tab["centre"], tab["lo"], tab["hi"], m)
        _u, de = PB.fit(A, tab, counts, U_target=target, ridge_rel=1e-3, ridge_form="eig")
        _u, dt = PB.fit(A, tab, counts, U_target=target, ridge_rel=1e-3, ridge_form="trace")
        lam[m] = (de, dt)
    # eig form: lambda / eig_max is the same constant at both m, by construction
    assert lam[8][0]["ridge_lambda"] / lam[8][0]["eig_max"] == pytest.approx(
        lam[64][0]["ridge_lambda"] / lam[64][0]["eig_max"], rel=1e-9)
    # trace form: the same ridge_rel buys a DIFFERENT fraction of eig_max at the two m
    r8 = lam[8][1]["ridge_lambda"] / lam[8][1]["eig_max"]
    r64 = lam[64][1]["ridge_lambda"] / lam[64][1]["eig_max"]
    assert r64 < 0.95 * r8, f"the trace ridge did not change with m: {r8:.3e} -> {r64:.3e}"


def test_trace_ridge_loses_an_order_of_magnitude_on_the_real_ensemble():
    """The measurement that motivates the eig form, on the data the project is about: on the
    dihedral's round-4 ensemble trace/m divided by eig_max is 0.430 at m=8 and 0.049 at m=128, so a
    ridge_rel tuned on a Chebyshev design regularises 8.8x less at the fine end. Skipped when
    results/plan_c is not in the checkout."""
    pc = REPO / "results" / "plan_c"
    field, ens = pc / "fields" / "c2stab" / "C2s8_r3.npz", pc / "ensembles_c2stab.npz"
    if not (field.exists() and ens.exists()):
        pytest.skip("results/plan_c is not present in this checkout")
    tab = L.load_field(field)["dihedral"]
    counts = np.load(ens)["C2s8_r4__dihedral"].astype(float)
    p = I.probability_from_counts(counts)
    ratios = {}
    for m in (8, 64, 128):
        A = PB.design_bspline(tab["centre"], float(tab["lo"]), float(tab["hi"]), m)
        Aw = A * np.sqrt(p)[:, None]
        G = Aw.T @ Aw
        ratios[m] = float(np.trace(G)) / m / float(np.linalg.eigvalsh(G)[-1])
    assert ratios[8] > 0.4 and ratios[128] < 0.08, ratios
    assert ratios[128] < ratios[8] / 5.0, f"the trace ridge is not m-dependent here: {ratios}"


def test_roughness_penalty_prefers_smooth_coefficients():
    """A P-spline penalty must leave the design alone and only shrink the coefficient oscillation."""
    tab, counts = _table()
    target = L._target_from_counts(counts)
    m = 32
    A = PB.design_bspline(tab["centre"], tab["lo"], tab["hi"], m)
    pen = PB.roughness_penalty(m)
    assert pen.shape == (m, m)
    _u0, d0 = PB.fit(A, tab, counts, U_target=target, ridge_rel=1e-6, ridge_form="eig")
    _u1, d1 = PB.fit(A, tab, counts, U_target=target, penalty=pen, penalty_rel=1e-2,
                     ridge_rel=1e-6, ridge_form="eig")
    rough0 = float(np.sum(np.diff(np.asarray(d0["coef"]), 2) ** 2))
    rough1 = float(np.sum(np.diff(np.asarray(d1["coef"]), 2) ** 2))
    assert rough1 < rough0, "the penalty did not smooth the coefficients"
