"""Plan C, first experiment: three arms, ONE seven-chain pool, native retention as the verdict.

docs/plan_c_selfconsistent_target.md:70 is the specification. The three arms differ in two things
and nothing else:

  C0 (control)  reference = the deposited pooled marginal (results/refit_smooth5.npz),
                fit = the loop's own marginal inversion (ibi_bonded.plan_update, the operator the
                production 867-chain run used for rounds 0-4)
  C1            reference = the ensemble the FIELD ITSELF produced from native starts,
                fit = a smoothed table:  U_{r+1} = smooth_U(-kBT ln p_ens_r)
  C2            same reference, fit = Chebyshev T_1..T_K:  U_{r+1} = proj_K(-kBT ln p_ens_r)

WHY C1/C2 REPLACE THE FIELD AND C0 INCREMENTS IT. Against a GENERATED reference the increment is a
no-op by construction: the target is what the simulation just produced, so kBT*ln(p_sim/p_ref) = 0
at round one and IBI would report convergence of a field that has not been refitted at all. The
self-consistent loop is therefore a FIT of the field to its own ensemble -- that is the "iteration
by construction" IsRNA2 does, and it is the whole content of the target change. C0 keeps the
increment because that is what today's loop does; it is the control, not a fourth idea.

WHY RETENTION AND NOT J. The 867-chain rounds were judged by J, and J's own floor on this grid is
0.0659 (docs/ibi_loop_and_oxrna_findings.md Part 5). The plan's acceptance is instead: a chain
relaxed and sampled under the final field stays near the geometry it was deposited in. The
instrument that measures that already exists and produced the baseline this experiment has to beat
-- scripts/measure_native_retention.py, 24 chains, 12 ps, tables_r3, 5000 relax steps: median
deposited -> sampled-mean RMSD 0.83 A, ensemble spread 0.22 A, 0 of 24 chains moved more than 10 A.
Rounds here REUSE that instrument unchanged (measure_native_retention._one) so the numbers are the
same measurement and not a second opinion; --baseline-tables re-runs it on the arms' own chains so
the 0.83 A comparison is made on the same chains rather than across two chain sets.

THE POOL IS THE A/B POOL. Seven chains, rebuilt by the same rule the A/B arms used
(ibi_loop.py:662, the "small" band): every loaded structure with >= 8 pairs and 24 <= L <= 34, in
the loader's alphabetical order, first seven. tests/test_plan_c_loop.py pins the names, because a
loader that reorders silently would change the pool and nothing else in this script would notice.

WHAT EACH ROUND RECORDS, and why each one is there:

  n_total / n_outside   the ensemble's own support check: outside mass is counted, never dropped
  max|dU|               how far the field moved this round, in kJ/mol and in kBT
  fit_implied_ln_ratio  C1/C2 only: max |ln(p_implied/p_target)| between the stored table read
                        back through the sampler's piecewise-linear convention
                        (ibi_bonded.bin_probabilities_from_U) and the ensemble it was fitted to.
                        The point-weight convention is what the fit inverts, and the sampler is
                        not that convention; this is the size of the difference on THIS grid, so
                        the next reader does not have to guess it.
  ensemble_vs_prev      the stationarity instrument the plan asks for (doc:64): mass-weighted mean
                        and max |ln(p_r/p_{r-1})| of the pooled ensemble between consecutive
                        rounds. C1's replacement map should drive this to sampling noise in one
                        round; whether C2's smooth field does is the experiment.
  dep                   sim/ref and the joint J against the DEPOSITED marginal, both denominators
                        (stored sigma and the table's own implied sigma, ibi_core.simref_table),
                        so this run is readable next to the 867-chain series
  retention             the acceptance number, pool chains and held-out chains separately

Run (six workers maximum, one torch thread each: the production 867-chain run holds the rest):
  python scripts/plan_c_loop.py --rounds 4 --nrep 6 --nsteps 5000
Smoke (wiring only, numbers are not results):
  python scripts/plan_c_loop.py --rounds 1 --nrep 1 --nsteps 200 --burn 40 --relax 200 \
      --pool-size 2 --holdout 1 --retention-steps 200 --retention-relax 200 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# THE PINS GO BEFORE torch IS IMPORTED. Measured 2026-09-21: a worker pool that imports torch first
# reserves 950 MB of commit per process on this box, against 181 MB with OMP/MKL pinned and
# KMP_BLOCKTIME at 0 (docs/plan_c_selfconsistent_target.md:174 -- this is the trap the
# native-retention launch fell into). setdefault, not assignment: a caller that pins differently on
# purpose must win.
for _name, _default in (("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("KMP_BLOCKTIME", "0")):
    os.environ.setdefault(_name, _default)

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B                          # noqa: E402
import cg_potentials as P                             # noqa: E402
import ibi_bonded as I                                # noqa: E402
import ibi_core as IC                                 # noqa: E402
import measure_native_retention as NR                 # noqa: E402
import torusfold.scheme2.torch_cgsim as C             # noqa: E402

OUT_ROOT = REPO / "results" / "plan_c"
DEFAULT_REF = REPO / "results" / "refit_smooth5.npz"
DEFAULT_BASELINE = REPO / "results" / "ibi_relax" / "tables_r3.npz"
# The same wall, friction, cap and coordinates the production loop runs with, so a round here
# differs from a production round in the pool, the length of the window and the update rule only.
WALL_K = 2000.0
FRICTION = 1.0
FORCE_CAP = 5000.0
UPDATED = ("bb_bond", "angle", "dihedral")
CARRIED = ("stack",)
# C2's default order. The plan says K=4-8 (doc:78); 8 is the same order the moment operator's A/B
# settled on, so a coefficient of C2's can be read beside one of Plan B's.
DEFAULT_K = 8
# Below this target probability a bin's |ln ratio| is the pseudo-count talking rather than an
# observation, and a maximum over such bins is meaningless. 1e-6 is one observation per 10^6 -- on
# the smallest ensemble here (2 chains x 1 replica x 4 frames in the smoke run) that is far inside
# the pseudo-count's own scale, and on the full pool it is two orders below one frame per bin.
LN_RATIO_FLOOR = 1e-6
# Plan C's second task: make the parametrised fit stable, or have a rule stop it and say so.
# RIDGE_REL is moment_correction's own relative ridge -- 1e-3 x trace/K on the same Chebyshev design
# (ibi_bonded, docs/plan_b_coupled_update.md) -- so C2s and Plan B's operator are fitted the same
# way and can be read beside each other. SUPPORT_FRAC drops the bins whose ensemble probability is
# below 1e-3 of the mode, which is where the target -kBT ln p is the logarithm of a pseudo-count.
# MEASURED, not assumed (scripts, 2026-09-22, on the real round-1 ensembles of the c2stab run).
# With moment_correction's own relative ridge of 1e-3 the refit's tail oscillates: the weighted
# Gram matrix has condition 25-347, so the small eigenvalues are barely determined, and a
# max|dU| of 61 kJ/mol (bb_bond) sat in bins the ensemble visits rarely. Sweeping the ridge at K=8:
#
#     ridge   lambda   U_fit min..max     max|dU|   applied std (bb/ang/dih)
#     1e-3    4.7e-4   -44.1 .. 11.1       60.6     1.69 / 1.09 / 4.63
#     1e-2    4.7e-3   -26.4 ..  6.5       42.9     1.98 / 1.28 / 4.53
#     1e-1    4.7e-2    -5.5 ..  5.3       23.2     2.58 / 2.09 / 4.08
#     1e+0    4.7e-1     0.8 ..  7.9       16.9     3.07 / 2.86 / 3.94
#
# The force-relevant size of the step (the mass-weighted std) barely moves across the sweep -- the
# ensemble really does disagree with the current field by 1-5 kJ/mol in the first round -- while
# max|dU| shrinks by a factor of 3.5 as the tail stops oscillating. 1e-1 is the knee: it is the
# smallest ridge whose U_fit stays within the range of the target it is fitting (the target's own
# span is 0-22 kJ/mol), so the fit is a fit rather than an extrapolation.
RIDGE_REL = 1e-1
SUPPORT_FRAC = 1e-3
# How far below the support cut the fitted correction is tapered to zero, in decades of p. 2 means
# the fit has full authority where the ensemble's probability is above the cut, none two decades
# below it, and a smoothstep in between.
TAPER_DECADES = 2.0


def _is_stable_c2(arm):
    """True for the stabilised Chebyshev arms: C2s, C2s2, C2s4, C2s8."""
    return arm == "C2s" or (arm.startswith("C2s") and arm[3:].isdigit())


def _arm_K(arm, default):
    """C2s4 means the stabilised fit at K=4; every other arm takes --K."""
    if arm.startswith("C2s") and arm[3:].isdigit():
        return int(arm[3:])
    return int(default)


# --------------------------------------------------------------------------- field files
def load_field(path):
    """The stored tables, six keys per coordinate (the format write_round_file produces)."""
    return I.load_clean_tables(str(path))


def write_field(path, tabs):
    """A file use_table_file and ibi_core.load_tables can BOTH read.

    The contract of ibi_loop.write_round_file (read there): lo/hi/binw/U/centre/sigma per
    coordinate, in ONE file, all six coordinates present even though only four are ever sampled.
    The two rigid coordinates are carried, not recomputed -- they are SHAKE constraints and
    nothing in this loop can move them.
    """
    payload = {}
    for c, t in tabs.items():
        payload[f"{c}__lo"] = float(t["lo"])
        payload[f"{c}__hi"] = float(t["hi"])
        payload[f"{c}__binw"] = float(t["binw"])
        payload[f"{c}__U"] = np.asarray(t["U"], dtype=float)
        payload[f"{c}__centre"] = np.asarray(t["centre"], dtype=float)
        payload[f"{c}__sigma"] = float(t["sigma"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


# --------------------------------------------------------------------------- the pool
def build_pool(n_pool, n_holdout, structs=None):
    """(pool, holdout) from the band the A/B arms used, in the loader's order.

    ibi_loop.py:662 is the rule: len(pairs) >= 8 and 24 <= L <= 34. Only the HEAD of that list is
    taken (the A/B arms took the same head), so the pool is a property of the loader, not of this
    script's arithmetic; the names are printed and pinned by a test for that reason.

    The holdout is the next n_holdout entries of the same band: same length class, never sampled
    into any arm's ensemble, never relaxed under a fitted field before it is scored.

    structs is injectable because the band the rule selects depends on WHICH DATABASE is mounted
    (the full _cgdata/combined set the production launcher points at has hundreds; a bare checkout
    has eight), and a rule that can only be exercised against the mounted one is a rule nobody
    tests.
    """
    if structs is None:
        structs = B.load_structures(limit=5000)
    small = [s for s in structs if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
    if len(small) < n_pool + n_holdout:
        raise SystemExit(f"the 24-34 band has {len(small)} chains, asked for "
                         f"{n_pool} + {n_holdout}")
    return small[:n_pool], small[n_pool:n_pool + n_holdout]


# --------------------------------------------------------------------------- workers
def _build_potentials(field_npz):
    """use_table_file FIRST, then make_potential: the potential closes over the current table.

    The same three lines as ibi_loop._build_potentials, duplicated on purpose: importing the
    production driver would run its module-level environment parsing and point this script's
    OUT_ROOT at the run that must not be touched. It stays INSIDE the worker because the closure
    cannot be pickled -- a worker handed only the field's path is one use_table_file() call away
    from sampling the wrong field.
    """
    P.use_table_file(str(field_npz))
    pots = []
    for coord in UPDATED:
        spec = P.resolve_spec(f"table_wall:{WALL_K:g}", coord) if coord == "bb_bond" \
            else P.resolve_spec("table", coord)
        pots.append((coord, spec, P.make_potential(coord, spec)))
    return pots, P.potential_kwargs(pots)


def _sample_one(task):
    """One chain, one round, in its own process: relax under the arm's field, then sample it.

    Returns COUNTS and the moments accumulator, not values: the update needs the histogram on the
    table's own bins, and the accumulator is what ibi_core.simref turns into the sim/ref numbers
    the 867-chain series is quoted in. Nothing about positions comes back here -- retention is a
    separate instrument (measure_native_retention) run under the SAME field file, and duplicating
    it here is how the two protocols would drift apart.
    """
    (idx, name, L, pos_np, pairs, field_npz, nrep, nsteps, burn, stride, blocks, seed,
     relax) = task
    torch.set_num_threads(1)
    t0 = time.time()
    _pots, pot_kw = _build_potentials(field_npz)
    tab = IC.load_tables(str(field_npz))
    pos = torch.tensor(np.asarray(pos_np, dtype=np.float64).reshape(1, 3 * L, 3)).repeat(nrep, 1, 1)
    ij = torch.tensor(np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
    res = IC.run_round(pos=pos, vel=torch.zeros_like(pos), ij=ij,
                       pw=torch.ones(len(ij), dtype=torch.float32),
                       temps=torch.full((nrep,), 300.0, dtype=torch.float64), tab=tab,
                       nsteps=nsteps, burn=burn, stride=stride, blocks=blocks, friction=FRICTION,
                       force_cap=FORCE_CAP, pot_kw=pot_kw, seed=seed, nrep=nrep, progress=False,
                       constraints=C.make_intra_constraints(L), relax=relax,
                       log=lambda *a, **k: None)
    return idx, {"name": name, "L": L,
                 "counts": {c: np.asarray(res.counts[c], dtype=np.int64) for c in B.COORDS},
                 "acc": {c: [float(v) for v in np.asarray(res.acc[c], dtype=float)]
                         for c in B.COORDS},
                 "n_total": {c: int(res.n_total[c]) for c in B.COORDS},
                 "n_outside": {c: int(res.n_outside[c]) for c in B.COORDS},
                 "relax": res.relax, "seconds": time.time() - t0}


def _retention_one(task):
    """measure_native_retention._one, unchanged, so the number IS the baseline's measurement."""
    return NR._one(task)


def _order_key(r):
    """The two workers of this script do not return the same shape, so the sort does not assume.

    _sample_one returns (idx, payload) -- the index is part of its contract with the parent, which
    indexes chains by it. measure_native_retention._one returns its row dict with 'idx' inside,
    because that instrument is also called from a main() that only wants the rows.
    """
    return r[0] if isinstance(r, tuple) else r["idx"]


def _pool_map(fn, tasks, workers):
    """Run tasks over a spawn pool of at most this many processes, results in task order."""
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as procs:
        out = list(procs.imap_unordered(fn, tasks))
    out.sort(key=_order_key)
    return out


# --------------------------------------------------------------------------- fits
def _target_from_counts(counts, pseudo=I.DEFAULT_PSEUDO):
    """-kBT ln p_ens on the table's own bins: the Boltzmann inverse of the generated ensemble.

    The pseudocount policy is ibi_bonded's, called rather than re-derived, so a target here and a
    target in plan_update cannot disagree about what p is. The minimum is subtracted because a
    constant has no force and the stored tables are all shifted that way.

    The SUPPORT IS NOT TOUCHED. lo/hi/centre stay the reference's: the sampler's wall
    (table_wall:2000) is pinned to those edges, and a field on a different grid could not be
    compared with the previous round's, which is the one thing this loop has to do every round.
    """
    p = I.probability_from_counts(np.asarray(counts, dtype=float), pseudo=pseudo)
    U = -B.KBT * np.log(p)
    return U - U.min()


def _chebyshev_fit(table, U_target, counts, K):
    """U_target projected onto T_1..T_K on the table's own bins, weighted by the ensemble.

    WEIGHTED, and that is a measurement decision rather than taste. The target is -kBT ln(p), and p
    at a bin the ensemble visited twice is (2 + 0.5)/(N + 0.5*nbins): its U is the logarithm of a
    pseudo-count, tens of kJ/mol tall, and an unweighted least squares would let those bins decide
    the potential while the sampler never visits them. With w = sqrt(p_ens) the fit is dominated
    where the chain actually lives. The basis is ibi_bonded._chebyshev_design -- the same one Plan
    B's moment operator uses -- so K=8 means the same eight functions in both plans.
    """
    A, x = I._chebyshev_design(table["centre"], table["lo"], table["hi"], K)
    p = I.probability_from_counts(np.asarray(counts, dtype=float), pseudo=I.DEFAULT_PSEUDO)
    w = np.sqrt(p)
    coef, *_ = np.linalg.lstsq(A * w[:, None], np.asarray(U_target, dtype=float) * w, rcond=None)
    U_fit = A @ coef
    U_fit = U_fit - U_fit.min()
    return U_fit, {"fit": "chebyshev", "K": int(K), "coef": [float(v) for v in coef]}


def _chebyshev_fit_stable(table, U_target, counts, K, ridge=RIDGE_REL,
                          support_frac=SUPPORT_FRAC, gain=1.0, taper_decades=TAPER_DECADES):
    """The same projection with the three things the plain one does not have.

    WHAT BROKE, measured 2026-09-22 on the plain fit's first round: max|dU| came back 52.6 / 44.6 /
    45.7 kJ/mol against the control's 8.0 / 2.9 / 5.6, and the diagnosis (docs, and the
    mass-weighted columns this file records) was that the projection had put its MINIMUM outside
    the region the chains visit: over the occupied bins U_new - U_old was very nearly a constant,
    +49 kJ/mol with a spread of 1.5 (1.1 angle, 4.7 dihedral). max|dU| was reporting a gauge choice,
    not a force. Three fixes, each with a precedent in this repository rather than invented here:

    1. SUPPORT. The target is -kBT ln p and p at a bin the ensemble visited twice is a pseudo-count:
       its U is tens of kJ/mol tall, and those bins are everywhere outside the mass region. Rows
       whose p is below support_frac x max(p) are dropped from the fit entirely (weight 0), which is
       the same reasoning as the sqrt(p) weighting above, taken to its limit. The BASIS IS NOT
       MASKED: the fitted polynomial is still evaluated on every bin, because masking the correction
       would put a step in U and a delta in its force.

    2. RIDGE. lambda = ridge x trace(A^T W A) / K, the relative form ibi_bonded.moment_correction
       uses on the same Chebyshev design (Plan B, docs/plan_b_coupled_update.md). Relative, because
       the absolute size of A^T W A follows the ensemble's total weight, which is a property of the
       run (nrep x frames) rather than of the problem.

    3. THE GAUGE IS FIXED UNDER THE ENSEMBLE'S OWN MASS, not at the global minimum: U_fit -=
     <U_fit>_p. A constant has no force anywhere, so removing it changes nothing the sampler does
       and makes max|dU| comparable with the other arms' instead of reporting where the polynomial
       happened to bottom out.

    gain damps the step, the loop's own remedy for a coupled fixed-point iteration (ibi_loop:862).
    """
    A, _x = I._chebyshev_design(table["centre"], table["lo"], table["hi"], K)
    p = I.probability_from_counts(np.asarray(counts, dtype=float), pseudo=I.DEFAULT_PSEUDO)
    support = p > support_frac * max(p.max(), 1e-300)
    w = np.sqrt(np.where(support, p, 0.0))
    Aw = A * w[:, None]
    AtWA = Aw.T @ Aw
    # THE RIGHT-HAND SIDE CARRIES w ONCE MORE, and getting that wrong is what the first version of
    # this function did. Minimising ||Aw c - w*U_target||_2 -- the plain fit's lstsq, and the
    # weighting its docstring describes -- has the normal equations (A^T W^2 A) c = A^T W^2 U_target
    # with W = diag(w). Aw.T @ U_target is A^T W U_target: one factor of w short, which puts the
    # pseudo-count bins back in at sqrt(p) instead of p. Measured 2026-09-22 on the real round-1
    # ensembles: that version returned coefficients of -720..+122 with the weighted Gram matrix's
    # eigenvalues at 0.005..1.73 (condition 25-347, perfectly regular), and the field it produced
    # wanted a 2167-4078 kJ/mol step -- refused as unrepresentable on all three coordinates. The
    # spectrum was never the problem; the residual was.
    rhs = Aw.T @ (np.asarray(U_target, dtype=float) * w)
    lam = ridge * float(np.trace(AtWA)) / max(K, 1)
    coef = np.linalg.solve(AtWA + lam * np.eye(K), rhs)
    U_fit = A @ coef
    U_old = np.asarray(table["U"], dtype=float)
    # WHERE THE ENSEMBLE HAS NO MASS, THE OLD FIELD STANDS. A refit is only defined where there is
    # data; outside it the polynomial is an extrapolation, and measured on the real round-1
    # ensembles it dives -- bb_bond's fit reaches -49 kJ/mol against a stored table whose own
    # minimum is 0.2 -- i.e. it digs a well nobody asked for in a tail the sampler can still reach.
    # The taper is a smoothstep in ln p over taper_decades decades below the support cut, so the
    # correction goes to zero outside the data with a continuous first derivative. A hard mask would
    # put a step in U and a delta function in its force instead.
    logp = np.log(np.clip(p, 1e-300, None))
    log_cut = float(np.log(max(support_frac * p.max(), 1e-300)))
    _width = max(taper_decades * float(np.log(10.0)), 1e-9)
    _t = np.clip((logp - (log_cut - _width)) / _width, 0.0, 1.0)
    taper = 0.5 * (1.0 - np.cos(np.pi * _t))
    dU = gain * taper * (U_fit - U_old)
    # THE GAUGE, UNDER THE ENSEMBLE'S OWN MASS: a constant added to U has no force anywhere, so
    # removing it changes nothing the sampler does and makes max|dU| mean the same thing in every
    # arm instead of reporting where the polynomial happened to bottom out.
    psum = float(p.sum())
    offset = float((p * dU).sum() / psum) if psum > 0 else 0.0
    dU = dU - offset
    U_new = U_old + dU
    _eig = np.linalg.eigvalsh(AtWA)[::-1]
    return U_new, {"fit": "chebyshev_ridge", "K": int(K), "ridge_rel": float(ridge),
                   "eig_max": float(_eig[0]), "eig_min": float(_eig[-1]),
                   "cond": float(_eig[0] / max(_eig[-1], 1e-300)),
                   "taper_decades": float(taper_decades),
                   "taper_below_one": int((taper < 1.0).sum()),
                   "support_frac": float(support_frac), "support_bins": int(support.sum()),
                   "n_bins": int(p.size), "gain": float(gain),
                   "ridge_lambda": float(lam), "mass_offset_removed": float(offset),
                   "coef": [float(v) for v in coef]}


def _fit_coord(arm, table, counts, n_total, n_outside, p_ref, K, ridge=RIDGE_REL,
               support_frac=SUPPORT_FRAC, gain=1.0, taper_decades=TAPER_DECADES):
    """(U_new, diagnostics) for one coordinate under one arm. Raises on a refused C0 update.

    A refusal is not caught here: plan_update refuses when the simulation left the support or the
    step is too large, and a loop that swallowed that would report a converged arm made of empty
    rounds. The caller records the refusal and the arm stops -- see run_arm.
    """
    counts = np.asarray(counts, dtype=float)
    n_outside = int(n_outside)
    n_total = int(n_total)
    if arm == "C0":
        hist = I.SimHistogram(counts=counts, n=n_total, n_outside=n_outside,
                              lo=float(table["lo"]), hi=float(table["hi"]),
                              nbins=len(table["U"]))
        # smooth_bins is +-bins while boltzmann_bonded.SMOOTH_WIDTH is a window WIDTH; the
        # conversion is ibi_loop.py:893's, kept identical so C0's update IS the production update.
        res = I.plan_update(table, hist, p_ref, gain=1.0,
                            smooth_bins=(B.SMOOTH_WIDTH - 1) // 2)
        diag = {"status": res.status, "reason": res.reason}
        for k, v in res.diagnostics.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                diag[k] = None if isinstance(v, float) and v != v else v
            elif isinstance(v, (list, tuple, np.ndarray)):
                diag[k] = [float(u) for u in np.asarray(v, dtype=float).ravel()]
        if not res.ok:
            # IBIRefusal carries the reason and the diagnostics; the caller records both rather
            # than a message string, because the diagnostics are what says WHICH check fired.
            raise I.IBIRefusal(res.status, res.reason, res.diagnostics)
        return np.asarray(res.table["U"], dtype=float), diag

    U_target = _target_from_counts(counts)
    if arm == "C1":
        U_new = B.smooth_U(U_target, B.SMOOTH_WIDTH)
        diag = {"fit": "table", "smooth_width": int(B.SMOOTH_WIDTH)}
    elif arm == "C2":
        U_new, diag = _chebyshev_fit(table, U_target, counts, K)
    elif _is_stable_c2(arm):
        # C2s / C2s4 / C2s2: the same target and the same Chebyshev design, fitted with the
        # support cut, the relative ridge and the mass gauge (see _chebyshev_fit_stable).
        U_new, diag = _chebyshev_fit_stable(table, U_target, counts, K, ridge=ridge,
                                            support_frac=support_frac, gain=gain,
                                            taper_decades=taper_decades)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    # The fit inverts a POINT-WEIGHT probability; the sampler interpolates the stored U linearly
    # between bin centres and clamps outside them (ibi_bonded.sample_segments, _sample). The two
    # agree only in the small-binw limit, and bin_probabilities_from_U is the convention-consistent
    # one -- so the difference is measured here instead of being assumed small.
    p_target = I.probability_from_counts(counts, pseudo=I.DEFAULT_PSEUDO)
    try:
        p_implied = I.bin_probabilities_from_U(dict(table, U=np.asarray(U_new, dtype=float)))
    except ValueError as exc:
        # A FITTED FIELD CAN LEAVE THE REPRESENTABLE RANGE, and that is a refusal rather than a
        # crash. Measured 2026-09-22 on the smoke protocol with the C2s gain at 30: the fitted U
        # spanned more than exp(-U/kBT) can hold and ibi_bonded raised from bin_probabilities_from_U,
        # which took the whole run down before any record was written. The same can happen to a
        # fit at gain 1 whose basis dives in a tail, so the check is converted here into the
        # refusal path every arm already has: the previous table is kept, the round is recorded
        # with the step that was wanted, and a C2s arm stops there (run_arm).
        raise I.IBIRefusal(
            "unrepresentable",
            f"the fitted U cannot be normalised ({exc}); the fit wanted "
            f"max|dU| = {np.abs(np.asarray(U_new, dtype=float) - np.asarray(table['U'], dtype=float)).max():.1f} "
            f"kJ/mol at K={K}", {}) from exc
    m = p_target > LN_RATIO_FLOOR
    ratios = np.abs(np.log(np.clip(p_implied[m], 1e-300, None) / p_target[m])) if m.any() \
        else np.zeros(1)
    diag["fit_implied_ln_ratio_max"] = float(ratios.max())
    diag["fit_implied_ln_ratio_median"] = float(np.median(ratios))
    return np.asarray(U_new, dtype=float), diag


# --------------------------------------------------------------------------- round bookkeeping
def _sum_histograms(rows):
    """Pool per-chain counts/moments/n. Summing counts is exact: the chains are independent."""
    counts = {c: np.zeros_like(np.asarray(rows[0]["counts"][c])) for c in B.COORDS}
    acc = {c: np.zeros(3, dtype=float) for c in B.COORDS}
    n_total = {c: 0 for c in B.COORDS}
    n_outside = {c: 0 for c in B.COORDS}
    for r in rows:
        for c in B.COORDS:
            counts[c] = counts[c] + r["counts"][c]
            acc[c] = acc[c] + np.asarray(r["acc"][c], dtype=float)
            n_total[c] += r["n_total"][c]
            n_outside[c] += r["n_outside"][c]
    return counts, acc, n_total, n_outside


def _stationarity(p_prev, p_now):
    """Mass-weighted mean and max |ln(p_now/p_prev)| over the bins the ensemble has mass in.

    This is the plan's convergence criterion (doc:64, the distributions stop moving between
    rounds), spelled as a ratio rather than a total-variation distance because every other number
    in this repository is already a ln-ratio. Bins whose two-round average probability is below
    LN_RATIO_FLOOR are excluded: their ratio is a pseudo-count divided by a pseudo-count.
    """
    if p_prev is None:
        return None
    a = np.asarray(p_prev, dtype=float)
    b = np.asarray(p_now, dtype=float)
    w = 0.5 * (a + b)
    m = w > LN_RATIO_FLOOR
    if not m.any():
        return {"mass_weighted_mean": float("nan"), "max": float("nan"), "n_bins": 0}
    lr = np.abs(np.log(np.clip(b[m], 1e-300, None) / np.clip(a[m], 1e-300, None)))
    # A second, unit-free reading of the same distance: total variation between the two rounds'
    # ensembles restricted to the mass-bearing bins and renormalised. The ln-ratio mean is the
    # instrument the doc's convergence criterion is written in; TV says how much probability moved,
    # which is what "the distributions stop moving" means when a single bin's ratio is large.
    aa, bb = a[m], b[m]
    tv = 0.5 * float(np.abs(bb - aa).sum() / max(0.5 * (aa.sum() + bb.sum()), 1e-300))
    return {"mass_weighted_mean": float((w[m] * lr).sum() / w[m].sum()),
            "max": float(lr.max()), "n_bins": int(m.sum()), "total_variation": tv}


def summarize_retention(rows):
    """median/mean deposited -> sampled-mean RMSD, spread, and how many chains left the deposit.

    The 10 A threshold is the baseline's own (docs/plan_c_selfconsistent_target.md:160): it is the
    line between "relaxed to a slightly different but recognisably native geometry" and "melted",
    and it is quoted as a count rather than a fraction so a small group cannot hide behind one.
    """
    if not rows:
        return None
    dep = np.asarray([r["rmsd_dep_mean"] for r in rows], dtype=float)
    spread = np.asarray([r["rmsd_frame_spread"] for r in rows], dtype=float)
    moved = [r["name"] for r in rows if r["rmsd_dep_mean"] > 10.0]
    return {"n": len(rows), "median_dep_mean": float(np.median(dep)),
            "mean_dep_mean": float(dep.mean()), "max_dep_mean": float(dep.max()),
            "median_spread": float(np.median(spread)),
            "n_moved_over_10A": len(moved), "moved": moved,
            "worst": [[float(v), n, int(L)] for v, n, L
                      in sorted(((r["rmsd_dep_mean"], r["name"], r["L"]) for r in rows),
                                reverse=True)[:3]]}


def run_retention(field_path, chains, seed, steps, relax, workers):
    """The baseline instrument on these chains under this field, summarized."""
    t0 = time.time()
    tasks = [(i, s["name"], len(s["pos"]), np.asarray(s["pos"], dtype=np.float64), s["pairs"],
              str(field_path), int(steps), int(relax), int(seed) + i)
             for i, s in enumerate(chains)]
    rows = list(_pool_map(_retention_one, tasks, workers))      # dicts, sorted by task index
    rows.sort(key=lambda r: r["L"])
    return {"summary": summarize_retention(rows), "rows": rows, "seconds": time.time() - t0,
            "field": str(field_path), "nsteps": int(steps), "relax": int(relax)}


def _num(v):
    """float or None; nan is not JSON."""
    if v is None:
        return None
    v = float(v)
    return None if v != v else v


# --------------------------------------------------------------------------- one arm
def run_arm(arm, start_tables, pool, holdout, args, p_ref, ens_store, flush, rec):
    """R rounds of one arm, filling the caller's record in place; fields are written as it goes.

    REC IS PASSED IN, not returned: the per-round flush has to see the arm's record while the arm
    is still running. Measured 2026-09-22 on the first full pass -- the caller installed the return
    value into the shared record only after the arm was over, so every flush wrote a document whose
    in-flight arm was absent and the LAST arm of the matrix never reached the file at all, while the
    log and the field files said everything had worked.

    A refused C0 update ends the arm rather than being papered over: the refusal carries the
    reason (support drift, step too large, divergence) and a loop that continued past it would
    report a converged control arm made of rounds that changed nothing.
    """
    field_dir = OUT_ROOT / "fields" / args.tag
    tables = {c: dict(t) for c, t in start_tables.items()}
    rec.clear()
    rec.update({"arm": arm, "rounds": [], "fields": [], "refusals": [], "n_holdout": len(holdout)})
    p_prev = {c: None for c in B.COORDS}
    # The applied step norm per coordinate, for the divergence guard on the fitted arms: the same
    # history ibi_loop keeps for Plan B's moment operator (ibi_loop:885) and rebuilt per arm.
    norm_hist = {c: [] for c in UPDATED}
    field_path = write_field(field_dir / f"{arm}_r0.npz", tables)
    rec["fields"].append(str(field_path))

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        tasks = [(i, s["name"], len(s["pos"]), np.asarray(s["pos"], dtype=np.float64), s["pairs"],
                  str(field_path), int(args.nrep), int(args.nsteps), int(args.burn),
                  int(args.stride), int(args.blocks), int(args.seed) + 1000 * rnd + i,
                  int(args.relax)) for i, s in enumerate(pool)]
        sampled = [r for _i, r in _pool_map(_sample_one, tasks, args.workers)]
        t_sample = time.time() - t0
        counts, acc, n_total, n_outside = _sum_histograms(sampled)
        # The entry state, per chain, every round. The plan's first free check (doc:107) measured
        # that the deposited state does NOT predict the residual (Spearman(J, E0) = +0.043), and a
        # self-consistent loop is the one place that could stop being true: a fit that tightens the
        # field shows up here first, as chains that start capped and never leave.
        entry = {r["name"]: {
            "L": int(r["L"]),
            "at_cap": None if not r["relax"] else bool(r["relax"]["hit_cap"]),
            "left_cap": None if not r["relax"] else bool(r["relax"]["left_cap"]),
            "relax_accepted": None if not r["relax"] else int(r["relax"]["accepted"]),
            "force_start": None if not r["relax"] else float(r["relax"]["max_force_start"]),
            "force_end": None if not r["relax"] else float(r["relax"]["max_force_end"]),
            "energy_start": None if not r["relax"] else float(r["relax"]["energy_start"]),
            "energy_end": None if not r["relax"] else float(r["relax"]["energy_end"]),
            "seconds": float(r["seconds"])} for r in sampled}

        new_tables = {}
        per_coord = {}
        refused = None
        for c in tables:
            tab = tables[c]
            if c not in UPDATED:
                new_tables[c] = dict(tab)
                per_coord[c] = {"carried": True}
                continue
            try:
                U_new, diag = _fit_coord(arm, tab, counts[c], n_total[c], n_outside[c],
                                         p_ref[c], _arm_K(arm, args.K), ridge=args.ridge,
                                         support_frac=args.support_frac, gain=args.c2_gain,
                                         taper_decades=args.taper_decades)
            except I.IBIRefusal as exc:
                refused = f"{c}: {exc}"
                diag = {"status": "refused", "reason": str(exc)}
                U_new = np.asarray(tab["U"], dtype=float)
            dU = np.asarray(U_new, dtype=float) - np.asarray(tab["U"], dtype=float)
            diag["max_dU"] = float(np.abs(dU).max())
            diag["max_dU_kbt"] = float(np.abs(dU).max() / B.KBT)
            # THE ZERO OF A FITTED POTENTIAL IS A GAUGE CHOICE, and max|dU| alone can be blind to
            # whether anything physical moved. Measured 2026-09-22 on C2's first round: the K=8
            # projection's range was 63 kJ/mol against the target's 23, its minimum landed outside
            # the region the chains visit, and over the OCCUPIED bins U_new - U_old was a constant
            # +49 (+41 angle, +37 dihedral) with a spread of 1.5 (1.1, 4.7) kJ/mol. A constant has
            # no force: retention under that field came back identical to the round before. So the
            # force-relevant size of a step is reported under the ensemble's OWN mass, where a
            # constant drops out, alongside the raw maximum.
            p_mass = I.probability_from_counts(counts[c])
            dU_mean = float((p_mass * dU).sum())
            diag["mass_weighted_abs_dU"] = float(np.abs(p_mass * dU).sum())
            diag["mass_weighted_std_dU"] = float(
                np.sqrt((p_mass * (dU - dU_mean) ** 2).sum()))
            diag["dU_offset_under_mass"] = dU_mean
            # THE GUARD ON THE FITTED ARMS. Plan B's coupled step is checked by
            # ibi_bonded.divergence_check on the moment norm before it is injected (ibi_loop:885),
            # and a fitted correction gets the same instrument on the FORCE-RELEVANT size of its
            # step -- the mass-weighted std, not max|dU|, which is blind to exactly the constant the
            # plain fit's first round was made of. div == True means the step has grown for
            # PATIENCE consecutive rounds AND is GROWTH times its value at the start of that
            # window, i.e. a fixed-point iteration walking away: the previous table is kept for
            # that coordinate, the step that was refused is still recorded in the diagnostics, and
            # the arm stops after this round rather than reporting rounds that changed nothing.
            if _is_stable_c2(arm):
                _div, _msg = I.divergence_check(norm_hist[c], diag["mass_weighted_std_dU"])
                if _div:
                    refused = f"{c}: {_msg}"
                    diag["status"] = "refused"
                    diag["reason"] = _msg
                    U_new = np.asarray(tab["U"], dtype=float)
                else:
                    norm_hist[c].append(diag["mass_weighted_std_dU"])
            per_coord[c] = diag
            new_tables[c] = dict(tab, U=np.asarray(U_new, dtype=float))
        tables = new_tables
        field_path = write_field(field_dir / f"{arm}_r{rnd}.npz", tables)
        rec["fields"].append(str(field_path))
        if refused:
            rec["refusals"].append({"round": rnd, "why": refused})

        # the ensemble, stored so the stationarity claim can be re-derived from disk
        for c in B.COORDS:
            ens_store[f"{arm}_r{rnd}__{c}"] = np.asarray(counts[c], dtype=np.int64)
        for c in UPDATED:
            ens_store[f"{arm}_r{rnd}__n_total__{c}"] = np.int64(n_total[c])
            ens_store[f"{arm}_r{rnd}__n_outside__{c}"] = np.int64(n_outside[c])

        # the deposited-marginal residual, both denominators, on the field that was just fitted
        vals, j = IC.simref(acc, tables, skip=tuple(B.CONSTRAINED))
        vals_t, j_t = IC.simref_table(acc, tables, skip=tuple(B.CONSTRAINED))
        dep = {"sim_ref": {c: _num(v) for c, v in zip(B.COORDS, vals)},
               "joint_J": _num(j),
               "sim_ref_table": {c: _num(v) for c, v in zip(B.COORDS, vals_t)},
               "joint_J_table": _num(j_t)}

        stat = {c: _stationarity(p_prev[c], I.probability_from_counts(counts[c]))
                for c in UPDATED}
        p_prev = {c: I.probability_from_counts(counts[c]) for c in B.COORDS}

        ret = None
        if args.retention:
            ret = {"pool": run_retention(field_path, pool, args.seed + 77, args.retention_steps,
                                         args.retention_relax, args.workers),
                   "holdout": run_retention(field_path, holdout, args.seed + 91,
                                            args.retention_steps, args.retention_relax,
                                            args.workers)}

        rec["rounds"].append({
            "round": rnd, "field": str(field_path), "seconds_sample": t_sample,
            "seconds_total": time.time() - t0, "n_chains": len(sampled),
            "n_total": {c: n_total[c] for c in B.COORDS},
            "n_outside": {c: n_outside[c] for c in B.COORDS},
            "outside_frac": {c: (n_outside[c] / n_total[c] if n_total[c] else float("nan"))
                             for c in B.COORDS},
            "fit": per_coord, "stationarity": stat, "dep": dep, "entry": entry,
            "retention": None if ret is None else {
                "pool": {"summary": ret["pool"]["summary"], "rows": ret["pool"]["rows"]},
                "holdout": {"summary": ret["holdout"]["summary"], "rows": ret["holdout"]["rows"]},
            },
        })
        r = rec["rounds"][-1]
        _jv = dep["joint_J"]
        _jv = float("nan") if _jv is None else float(_jv)
        print(f"  {arm} r{rnd}: {t_sample:.0f} s sample, "
              + " ".join(f"{c} max|dU|={per_coord[c].get('max_dU', float('nan')):.3f}"
                         for c in UPDATED)
              + f"  J_dep={_jv:.4f}"
              + (f"  REFUSED {refused}" if refused else ""))
        # The force-relevant size of the step beside the raw maximum: the two disagree by a factor
        # of thirty on the plain fit's first round (52 kJ/mol of max against 1.5 of mass-weighted
        # std), and the second column is the one a sampler can feel.
        print("      mass-weighted dU (sum / std, kJ/mol): "
              + " ".join(f"{c} {per_coord[c].get('mass_weighted_abs_dU', float('nan')):.2f}/"
                         f"{per_coord[c].get('mass_weighted_std_dU', float('nan')):.2f}"
                         for c in UPDATED))
        if stat.get("bb_bond"):
            # The self-consistency reading: how far this round's ensemble moved from the previous
            # one, per coordinate, in the ln-ratio the plan's convergence criterion is written in
            # plus the total-variation distance between the same two histograms.
            print("      stationarity |ln p_r/p_{r-1}| mean/max/TV: " + " ".join(
                f"{c} {stat[c]['mass_weighted_mean']:.3f}/{stat[c]['max']:.2f}/"
                f"{stat[c].get('total_variation', float('nan')):.3f}"
                if stat.get(c) else f"{c} n/a" for c in UPDATED))
        if r["retention"]:
            for grp in ("pool", "holdout"):
                sm = r["retention"][grp]["summary"]
                print(f"      retention {grp:7s} n={sm['n']} median {sm['median_dep_mean']:.2f} A "
                      f"spread {sm['median_spread']:.2f} A moved>10A {sm['n_moved_over_10A']}")
        flush()
        if refused and _is_stable_c2(arm):
            # A fitted arm stops on its first refusal. The step it wanted is already in the record
            # (the diagnostics above are of the REFUSED step, not of the table that was kept), and
            # carrying on would run the remaining rounds under a field the guard just rejected.
            print(f"      arm {arm} stops: the fitted step was refused at round {rnd}")
            break
    return rec


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Plan C arms C0/C1/C2 on the seven-chain pool")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--nrep", type=int, default=6)
    ap.add_argument("--nsteps", type=int, default=5000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--relax", type=int, default=5000)
    ap.add_argument("--retention-steps", type=int, default=6000,
                    help="6000 x 0.002 ps = 12 ps, the baseline instrument's window")
    ap.add_argument("--retention-relax", type=int, default=5000)
    ap.add_argument("--retention", type=int, default=1, help="0 skips the acceptance measurement")
    ap.add_argument("--preflight-retention", type=int, default=1,
                    help="1 measures the start field and the tables_r3 baseline before the arms; "
                         "0 skips both, for a re-run whose preflight numbers are already on disk")
    ap.add_argument("--pool-size", type=int, default=7)
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6,
                    help="hard cap 6: the production 867-chain run owns the rest of the box")
    ap.add_argument("--arms", default="C0,C1,C2")
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--ridge", type=float, default=RIDGE_REL,
                    help="relative ridge for the C2s arms: lambda = ridge x trace(A^T W A) / K, "
                         "the same relative form ibi_bonded.moment_correction uses")
    ap.add_argument("--support-frac", type=float, default=SUPPORT_FRAC,
                    help="bins below this fraction of the modal ensemble probability are dropped "
                         "from the C2s fit (their target is a pseudo-count's logarithm)")
    ap.add_argument("--c2-gain", type=float, default=1.0,
                    help="damping on the C2s step; 1.0 is the undamped fit")
    ap.add_argument("--taper-decades", type=float, default=TAPER_DECADES,
                    help="decades of p below the support cut over which the C2s correction is "
                         "tapered to zero, so the refit cannot act where the ensemble has no mass")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--baseline-tables", default=str(DEFAULT_BASELINE),
                    help="the field the 0.83 A baseline was measured under; pass '' to skip")
    ap.add_argument("--tag", default="run1")
    args = ap.parse_args(argv)
    if args.workers > 6:
        raise SystemExit("--workers above 6 would compete with the production run; that is a "
                         "decision for the operator, not a default")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    pool, holdout = build_pool(args.pool_size, args.holdout)

    ref_tables = load_field(args.ref)
    p_ref = {c: I.probability_from_table(ref_tables[c]) for c in B.COORDS}
    # field_0 is shared by every arm, so a difference between arms is the UPDATE and never the
    # starting point. It is the deposited marginal's own inversion (refit_smooth5), which is the
    # same field the production 867-chain run calls tables_r0.npz.
    start_tables = {c: dict(ref_tables[c]) for c in ref_tables}
    start_path = write_field(OUT_ROOT / "fields" / args.tag / "start.npz", start_tables)

    print(f"plan C {args.tag}: arms {arms}, {args.rounds} rounds, pool {len(pool)}, "
          f"holdout {len(holdout)}")
    print(f"  start field  {start_path}  (from {args.ref})")
    print("  pool   " + ", ".join(f"{s['name']}(L={len(s['pos'])})" for s in pool))
    print("  held   " + ", ".join(f"{s['name']}(L={len(s['pos'])})" for s in holdout))
    print(f"  rounds: {args.nrep} replicas x {args.nsteps} steps ({args.nsteps * 0.002:.0f} ps), "
          f"burn {args.burn} ({args.burn * 0.002:.0f} ps), stride {args.stride}, "
          f"relax {args.relax}; retention {args.retention_steps} steps "
          f"({args.retention_steps * 0.002:.0f} ps), relax {args.retention_relax}")
    print(f"  {args.workers} worker processes x 1 torch thread; seed base {args.seed}; "
          f"K={args.K}; C2s ridge/support/gain {args.ridge:g}/{args.support_frac:g}/{args.c2_gain:g}")

    record = {"tag": args.tag, "argv": list(argv) if argv is not None else sys.argv[1:],
              "created": datetime.now().isoformat(timespec="seconds"),
              "rounds": args.rounds, "nrep": args.nrep, "nsteps": args.nsteps,
              "burn": args.burn, "stride": args.stride, "blocks": args.blocks,
              "relax": args.relax, "workers": args.workers, "seed": args.seed, "K": args.K,
              "ridge": args.ridge, "support_frac": args.support_frac, "c2_gain": args.c2_gain,
              "taper_decades": args.taper_decades,
              "arm_K": {a: _arm_K(a, args.K) for a in arms},
              "ref": str(args.ref), "start_field": str(start_path),
              "baseline_tables": args.baseline_tables or None,
              "retention_protocol": {"nsteps": args.retention_steps,
                                     "relax": args.retention_relax, "stride": 25, "burn": 0,
                                     "instrument": "scripts/measure_native_retention.py"},
              "pool": [{"name": s["name"], "L": len(s["pos"])} for s in pool],
              "holdout": [{"name": s["name"], "L": len(s["pos"])} for s in holdout],
              "arms": {}}
    out_json = OUT_ROOT / f"plan_c_{args.tag}.json"
    ens_store = {}

    def flush():
        """Write after every arm-round: this is a multi-hour job and a crash must not eat it."""
        out_json.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
        np.savez(OUT_ROOT / f"ensembles_{args.tag}.npz", **ens_store)

    if args.retention and args.preflight_retention:
        print("\nretention of the SHARED start field (every arm starts here)")
        record["start_retention"] = {
            "pool": run_retention(start_path, pool, args.seed + 77, args.retention_steps,
                                  args.retention_relax, args.workers),
            "holdout": run_retention(start_path, holdout, args.seed + 91,
                                     args.retention_steps, args.retention_relax, args.workers)}
        for grp in ("pool", "holdout"):
            sm = record["start_retention"][grp]["summary"]
            print(f"  start/{grp:7s} median {sm['median_dep_mean']:.2f} A "
                  f"spread {sm['median_spread']:.2f} A moved>10A {sm['n_moved_over_10A']}")
        flush()
    if args.retention and args.preflight_retention:
        if args.baseline_tables:
            print(f"\nretention under the baseline field {Path(args.baseline_tables).name} "
                  f"(the field the 0.83 A number was measured under), same chains")
            record["baseline_retention"] = {
                "tables": str(args.baseline_tables),
                "pool": run_retention(args.baseline_tables, pool, args.seed + 77,
                                      args.retention_steps, args.retention_relax, args.workers),
                "holdout": run_retention(args.baseline_tables, holdout, args.seed + 91,
                                         args.retention_steps, args.retention_relax, args.workers)}
            for grp in ("pool", "holdout"):
                sm = record["baseline_retention"][grp]["summary"]
                print(f"  baseline/{grp:7s} median {sm['median_dep_mean']:.2f} A "
                      f"spread {sm['median_spread']:.2f} A moved>10A {sm['n_moved_over_10A']}")
            flush()

    for arm in arms:
        print(f"\n=== arm {arm} ===")
        record["arms"][arm] = {}            # installed BEFORE the arm runs: see run_arm's docstring
        run_arm(arm, start_tables, pool, holdout, args, p_ref, ens_store, flush,
                record["arms"][arm])
    flush()                                 # the last arm's record, on its own

    print("\n=== summary: deposited -> sampled-mean RMSD ===")
    print(f"{'arm':>6} {'round':>5} {'pool med':>9} {'pool moved':>11} "
          f"{'held med':>9} {'held moved':>11}")
    if args.retention and "start_retention" in record:
        a = record["start_retention"]["pool"]["summary"]
        b = record["start_retention"]["holdout"]["summary"]
        print(f"{'start':>6} {0:>5} {a['median_dep_mean']:>9.2f} "
              f"{a['n_moved_over_10A']:>4}/{a['n']:<5} {b['median_dep_mean']:>9.2f} "
              f"{b['n_moved_over_10A']:>4}/{b['n']:<5}")
        if record.get("baseline_retention"):
            a = record["baseline_retention"]["pool"]["summary"]
            b = record["baseline_retention"]["holdout"]["summary"]
            print(f"{'base':>6} {0:>5} {a['median_dep_mean']:>9.2f} "
                  f"{a['n_moved_over_10A']:>4}/{a['n']:<5} {b['median_dep_mean']:>9.2f} "
                  f"{b['n_moved_over_10A']:>4}/{b['n']:<5}")
    for arm in arms:
        for rnd in record["arms"][arm]["rounds"]:
            ret = rnd.get("retention")
            if not ret:
                continue
            a = ret["pool"]["summary"]
            b = ret["holdout"]["summary"]
            print(f"{arm:>6} {rnd['round']:>5} {a['median_dep_mean']:>9.2f} "
                  f"{a['n_moved_over_10A']:>4}/{a['n']:<5} {b['median_dep_mean']:>9.2f} "
                  f"{b['n_moved_over_10A']:>4}/{b['n']:<5}")
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
