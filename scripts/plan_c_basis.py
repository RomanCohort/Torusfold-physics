"""The correction families, one shared fit, and the diagnostics that decide which of them to sample.

THE PROJECT. Measured 2026-09-24 (docs/archive/plan_c_c2_stabilization.md section 5): on ONE coordinate, ONE
ensemble, at gain 1.0, a smooth global K=8 Chebyshev refit cycles -- corr(step_r, step_{r-1}) -0.94,
amplitude growing 2.72 -> 3.42 -> 4.06 kJ/mol, 13.8x the same-field floor, edge mass stuck at 0.43 --
while the 1000-bin table inversion converges and is the only rule that MOVES the edge mass (0.495 ->
0.301). Damping the refit to gain 0.5 or 0.2 also converges, but that buys stability by taking smaller
steps, not by being able to represent the target: the dihedral's marginal is bimodal with 31 per cent
of its mass in the outer five per cent of the support, and a global basis can only shuttle that mass
between the two edge regions. The question this module exists for: is there a family that is low
dimensional, smooth, AND able to carry that shape at full strength.

THE AXIS IS LOCALITY, and the two known outcomes are its ends: K=8 Chebyshev (global, cycles) and the
1000-bin table (fully local, converges). In between sits a one-parameter family of B-splines with m
basis functions; m* -- the knot count at which the cycle disappears -- is the answer, and it is also
what Plan B's moment operator should be using, since it fits every coordinate with the same K=8
Chebyshev and will hit the same wall the moment its target becomes a self-consistent ensemble.

WHAT IS SHARED AND WHAT IS NOT. The fit is the pipeline the C2s arms already use -- support cut at
support_frac x modal probability, sqrt(p) weighting, a smoothstep taper in ln p below the cut, the
mass gauge, and a ridge in moment_correction's relative form -- so a family's numbers are comparable
with the arms' own. Only the DESIGN MATRIX changes. tests/test_plan_c_basis.py pins that the Chebyshev
path reproduces plan_c_loop._chebyshev_fit_stable exactly; a silent difference there would invalidate
every comparison in the project.

A NOTE ON A PERIODIC BASIS, since the dihedral invites one: it is NOT available here. The dihedral's
support is [-1.040, 1.040] radians -- 2.08 rad, only 0.331 of a full turn -- so the window is a slice
of the circle and a periodic basis would assert continuity across a 4.2 rad gap nobody sampled.
Fourier on this window would be a global basis like Chebyshev, not a different idea.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B          # noqa: E402
import ibi_bonded as I                # noqa: E402

KBT = B.KBT
LN_FLOOR = 1e-6


# --------------------------------------------------------------------------- designs
def design_chebyshev(centre, lo, hi, K):
    """The reference: T_1..T_K on the support mapped to [-1, 1] (ibi_bonded._chebyshev_design)."""
    A, _x = I._chebyshev_design(np.asarray(centre, float), lo, hi, K)
    return A


def design_bspline(centre, lo, hi, m, degree=3):
    """m B-spline basis functions of the given degree over [lo, hi], uniform open knots.

    LOCALITY IS THE POINT: each basis function is nonzero over about (degree+1)/m of the support, so
    m=8 spans it and m=64 cannot see more than a sixteenth of it. That is the parameter the project
    sweeps; the knot count IS the axis, not an implementation detail.
    """
    from scipy.interpolate import BSpline
    centre = np.asarray(centre, float)
    m = int(m)
    if m <= degree:
        raise ValueError(f"m={m} must exceed the degree {degree}")
    n_inner = m - degree - 1
    inner = np.linspace(lo, hi, n_inner + 2)[1:-1] if n_inner > 0 else np.array([])
    knots = np.concatenate([[lo] * (degree + 1), inner, [hi] * (degree + 1)])
    cols = []
    for i in range(m):
        c = np.zeros(m)
        c[i] = 1.0
        cols.append(BSpline(knots, c, degree, extrapolate=False)(centre))
    A = np.nan_to_num(np.stack(cols, axis=1))
    return A


def design_table(centre, lo, hi, m=None):
    """The fully local limit: no basis at all, one free value per bin (m = nbins)."""
    n = len(np.asarray(centre))
    return np.eye(n)


def design(name, centre, lo, hi, m=8):
    """(design matrix, description) for a family name: cheb8, bspline32, pspline32, table."""
    name = str(name).strip().lower()
    if name.startswith("cheb"):
        k = int(name[4:]) if name[4:].isdigit() else int(m)
        return design_chebyshev(centre, lo, hi, k), f"chebyshev K={k}"
    if name.startswith("pspline"):
        mm = int(name[7:]) if name[7:].isdigit() else int(m)
        return design_bspline(centre, lo, hi, mm), f"pspline m={mm}"
    if name.startswith("bspline"):
        mm = int(name[7:]) if name[7:].isdigit() else int(m)
        return design_bspline(centre, lo, hi, mm), f"bspline m={mm}"
    if name == "table":
        return design_table(centre, lo, hi), "table (one value per bin)"
    raise ValueError(f"unknown family {name!r}")


def roughness_penalty(m, order=2):
    """D^T D for a finite-difference operator of the given order on the coefficient vector.

    The P-spline penalty: it does not change what the family CAN represent (the basis is unchanged),
    it changes what it PREFERS, which is what a family with more knots than its data needs.
    """
    m = int(m)
    D = np.eye(m)
    for _ in range(int(order)):
        D = np.diff(D, axis=0)
    return D.T @ D


# --------------------------------------------------------------------------- the shared fit
def fit(design_matrix, table, counts, U_target=None, ridge_rel=1e-1, support_frac=1e-3,
        taper_decades=2.0, gain=1.0, penalty=None, penalty_rel=0.0, pseudo=I.DEFAULT_PSEUDO,
        target_fn=None, ridge_form="eig"):
    """The C2s pipeline with a pluggable design. Returns (U_new, diagnostics).

    Support cut -> sqrt(p) weights -> ridge (lambda = ridge_rel x trace(A^T W A)/K, moment_correction's
    relative form) plus an optional coefficient roughness penalty -> smoothstep taper in ln p two
    decades below the cut -> mass gauge. The Chebyshev path is numerically identical to
    plan_c_loop._chebyshev_fit_stable (pinned by test_chebyshev_path_matches_the_loop).
    """
    A = np.asarray(design_matrix, float)
    n, m = A.shape
    centre = np.asarray(table["centre"], float)
    if centre.size != n:
        raise ValueError(f"design has {n} rows, the table has {centre.size} bins")
    counts = np.asarray(counts, float)
    if target_fn is not None:
        U_target = target_fn(counts)
    if U_target is None:
        raise ValueError("pass U_target or target_fn")
    U_target = np.asarray(U_target, float)

    p = I.probability_from_counts(counts, pseudo=pseudo)
    support = p > support_frac * max(p.max(), 1e-300)
    w = np.sqrt(np.where(support, p, 0.0))
    Aw = A * w[:, None]
    G = Aw.T @ Aw
    rhs = Aw.T @ (U_target * w)
    eig0 = float(np.linalg.eigvalsh(G)[-1])
    # RIDGE SCALE, measured. moment_correction's relative form is lambda = ridge_rel x trace(G)/m,
    # which for Chebyshev designs is about ridge_rel x eig_max/2 and therefore behaves like an
    # eigenvalue-relative ridge. For B-splines it is NOT, and the measurement is on the dihedral's own
    # round-4 ensemble (2026-09-24):
    #
    #     m      8      16      32      64      128
    #     trace/m  0.0607  0.0288  0.0145  0.00736 0.00373
    #     eig_max  0.141   0.0925  0.0845  0.0825  0.0761
    #     ratio    0.430   0.312   0.172   0.0891  0.0490
    #
    # trace/m falls by a factor of 16 across the sweep while eig_max falls by 1.9, so the SAME
    # ridge_rel regularises 8.8x less at m=128 than at m=8: the ridge tuned on a Chebyshev design
    # stops doing anything exactly where the design needs it, and by m=48 the weighted system is
    # numerically singular (the solve returned values past 1e100). eig_max is the scale that means
    # the same thing at every m, so it is the default here; "trace" is kept for the comparison with
    # the arms, which use it through plan_c_loop.
    scale = eig0 if ridge_form == "eig" else float(np.trace(G)) / max(m, 1)
    lam = ridge_rel * scale
    H = G + lam * np.eye(m)
    if penalty is not None and penalty_rel:
        # Same lesson for the roughness penalty: scaled by trace/m it is invisible at large m.
        H = H + penalty_rel * scale * np.asarray(penalty, float)
    coef = np.linalg.solve(H, rhs)
    U_fit = A @ coef

    logp = np.log(np.clip(p, 1e-300, None))
    log_cut = float(np.log(max(support_frac * p.max(), 1e-300)))
    width = max(taper_decades * float(np.log(10.0)), 1e-9)
    t = np.clip((logp - (log_cut - width)) / width, 0.0, 1.0)
    taper = 0.5 * (1.0 - np.cos(np.pi * t))
    U_old = np.asarray(table["U"], float)
    dU = gain * taper * (U_fit - U_old)
    psum = float(p.sum())
    offset = float((p * dU).sum() / psum) if psum > 0 else 0.0
    dU = dU - offset
    U_new = U_old + dU

    eig = np.linalg.eigvalsh(G)[::-1]
    # The condition number of the PENALISED system, i.e. of the matrix actually inverted: the design's
    # own condition number is a property of the basis, not of the fit.
    _hp = np.linalg.eigvalsh(H)
    cond_fit = float(_hp[-1] / max(_hp[0], 1e-300))
    # The residual is measured where the mass is, with the same gauge, BEFORE the taper can hide it:
    # this is the number that says whether a family can express the target, as opposed to whether the
    # step it happens to want is small.
    mask = p > LN_FLOOR
    diff = U_fit[mask] - U_target[mask]
    diff = diff - float((p[mask] * diff).sum() / p[mask].sum())
    resid = float(np.sqrt((p[mask] * diff ** 2).sum()))
    return U_new, {"family_n_basis": int(m), "ridge_lambda": float(lam),
                   "ridge_rel": float(ridge_rel), "penalty_rel": float(penalty_rel),
                   "resid_rms_kJ": resid, "cond": float(eig[0] / max(eig[-1], 1e-300)),
                   "cond_fit": cond_fit, "ridge_form": str(ridge_form), "ridge_scale": float(scale),
                   "eig_max": float(eig[0]), "eig_min": float(eig[-1]),
                   "support_bins": int(support.sum()), "taper_below_one": int((taper < 1.0).sum()),
                   "mass_offset_removed": float(offset), "gain": float(gain),
                   "max_abs_dU": float(np.abs(dU).max()),
                   "mass_weighted_std_dU": float(np.sqrt((p * (dU - (p * dU).sum()) ** 2).sum())),
                   "coef": [float(v) for v in coef]}


# --------------------------------------------------------------------------- diagnostics
def shape_corr(a, b, p):
    """Mass-weighted shape correlation of two steps over the bins the ensemble visits."""
    m = np.asarray(p) > LN_FLOOR
    if not m.any():
        return float("nan")
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    aa = a[m] - float((p[m] * a[m]).sum() / p[m].sum())
    bb = b[m] - float((p[m] * b[m]).sum() / p[m].sum())
    den = float(np.sqrt((p[m] * aa ** 2).sum() * (p[m] * bb ** 2).sum()))
    return float((p[m] * aa * bb).sum() / den) if den > 0 else float("nan")


def edge_mass(p, centre, lo, hi, frac=0.05):
    """The mass in the outer frac of the support at each end: the shape a global basis cannot move."""
    edge = frac * (hi - lo)
    c = np.asarray(centre, float)
    return float(np.asarray(p)[(c <= lo + edge) | (c >= hi - edge)].sum())


def implied_edge(table, U, frac=0.05):
    """The edge mass of the distribution a field implies (the family's own claim about the edges)."""
    pi = I.bin_probabilities_from_U(dict(table, U=np.asarray(U, float)))
    return edge_mass(pi, table["centre"], float(table["lo"]), float(table["hi"]), frac)


def diagnose(table, counts, U_new, U_old, U_cheb=None, diag=None, frac=0.05):
    """The phase-1 columns: representability, conditioning, the step, and the edges."""
    p = I.probability_from_counts(np.asarray(counts, float), pseudo=I.DEFAULT_PSEUDO)
    dU = np.asarray(U_new, float) - np.asarray(U_old, float)
    out = dict(diag or {})
    out["step_rms_kJ"] = float(np.sqrt((p * (dU - (p * dU).sum()) ** 2).sum()))
    out["edge_target"] = edge_mass(p, table["centre"], float(table["lo"]), float(table["hi"]), frac)
    out["edge_implied_new"] = implied_edge(table, U_new, frac)
    out["edge_implied_old"] = implied_edge(table, U_old, frac)
    out["edge_gap"] = out["edge_implied_new"] - out["edge_target"]
    if U_cheb is not None:
        dc = np.asarray(U_cheb, float) - np.asarray(U_old, float)
        out["corr_with_cheb"] = shape_corr(dU, dc, p)
    return out
