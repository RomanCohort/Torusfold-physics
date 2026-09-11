"""Iterative Boltzmann inversion: the table update, and the ways it can fail.

Direct Boltzmann inversion (boltzmann_bonded.py) builds U(q) = -kBT ln P_ref(q) from a
structure database one coordinate at a time, and it is exact only when the coordinates are
independent. The review this project follows states the weakness plainly -- DBI "does not
account for correlations between different degrees of freedom" -- and scripts/ibi_round0.py
measures the consequence: the marginal of a coupled chain is not the reference marginal, so
U = -kBT ln P_ref is not the potential of mean force of the simulated system.

Iterative Boltzmann inversion closes that gap by making the inversion a fixed point. With
U_i the potential used in round i and P_sim,i the binned marginal it produces,

    U_{i+1}(q) = U_i(q) + kBT * ln( P_sim,i(q) / P_ref(q) )

which is stationary exactly when the simulation reproduces the reference. Two conventions have
to be stated before anything is computed, because the fixed point depends on them:

  * the bins are the table's own bins. P_ref and P_sim are both bin probabilities on the
    same [lo, hi] by nbins grid, so the log ratio is defined bin by bin and the update is a
    vector of length nbins, one entry per bin centre;
  * U is a piecewise-linear function of q, not a set of point weights. A table that is
    sampled (boltzmann_bonded._sample) and a table that is histogrammed are the same
    function, so the stationary table is the piecewise-linear potential whose *bin
    integrals* match P_ref. This matters at the 0.1 kBT level: point weights exp(-U_j/kBT)
    and the bin integral of the same table differ by kBT * ln[ (1 - e^-x) / x ] where
    x = (U_{j+1} - U_j)/kBT, about kBT*x/2 per bin. See bin_probabilities_from_U().

This module is the update half only. Given the table that was simulated, a histogram of the
sampled coordinate on its bins, and the reference probability on those bins, it returns the
next table in the format boltzmann_bonded.prepare() expects, plus the diagnostics that say
whether that table should be used at all.

Failure modes are named, measured, and acted on rather than smoothed away:

  * a bin the simulation never visited carries no measurement of P_sim, only the
    pseudocount, and the pseudocount is a policy rather than evidence. Such bins are frozen
    at their previous value, never given a fabricated correction; if they hold more than
    max_unsupported_mass of the reference probability the round is refused and no table is
    returned at all;
  * a bin the simulation visits where the reference has no mass would need an infinite
    correction: refused;
  * support drift: if more than max_outside_frac of the samples fall outside [lo, hi], the
    histogram is a truncated view of the simulation and the correction is biased: refused.
    Widening the support is not something an update can do, it needs reference data;
  * divergence: one round's correction is capped in size, and a correction that keeps
    growing across rounds is refused rather than iterated further;
  * a refusal returns table=None. There is no partially usable table that quietly looks
    fine; require_table() turns the refusal into an exception.

Two things the machinery structurally cannot do, stated here so they are not discovered later
as surprises:

  * it cannot invent support for a region the simulation never visits. Freezing and refusing
    are the only honest options; interpolating across a gap asserts data that do not exist;
  * "converged" means the correction fell below the tolerance. It does not mean the answer is
    right. A simulator with a systematic error that is stationary under the update gives a
    fixed point that is self-consistent and wrong, and nothing computed from these numbers
    can tell that apart from a correct one. Only an external check can, which is why
    tests/test_ibi_bonded.py runs the update against a case whose answer it knows.

The table's geometry is exposed with it, because a sampler and the reference that checks it
must not disagree about what the table means: sample_segments() returns the piecewise-linear
function boltzmann_bonded._sample evaluates as a segment list (flat clamped ends included),
segment_moments() integrates exp(-U/kBT) and q*exp(-U/kBT) over part of a segment analytically,
and bin_probabilities_from_U() is built on those. An exact sampler built on the same two calls
samples the distribution this module calls the reference, by construction.

Wiring. scripts/ibi_round0.py already computes kBT*ln(P_sim/P_ref) from its sampler but never
applies it. A driver for it is four calls, and the loop body is:

    tables = ibi_bonded.load_clean_tables(REPO / "results" / "boltzmann_tables_clean.npz")
    p_ref = {c: ibi_bonded.probability_from_table(tables[c]) for c in B.COORDS}
    ... sample with the tables as they are, collect values[c] ...
    res = ibi_bonded.advance_from_samples(tables[c], values[c], p_ref[c],
                                          history=max_dU_history)
    if res.ok:
        tables[c] = res.table            # feed to B.prepare() and the next round
        max_dU_history.append(res.max_abs_dU)
    else:
        ... res.status, res.reason, res.diagnostics ...

p_ref must be kept from round 0 and never recomputed from the running table: taking it from
the table being updated would make the correction identically zero and report a converged run
that never converged.

Module only; the synthetic proof and the policy tests live in tests/test_ibi_bonded.py.
"""
from typing import NamedTuple

import numpy as np

try:
    import boltzmann_bonded as B
except ImportError as exc:  # environment error, not a data path
    raise ImportError(
        "ibi_bonded imports boltzmann_bonded, which sits beside it in scripts/; put "
        "scripts/ on sys.path before importing this module, as scripts/ibi_round0.py does"
    ) from exc

KBT = B.KBT

# Pseudocount policy. boltzmann_bonded._table_from_values uses the same value, and the update
# has to use the same one or the fixed point moves: it is a bias, not just a regulariser.
DEFAULT_PSEUDO = 0.5

# Per-bin floor for a usable measurement. With pseudo=0.5 the pseudocount carries 0.5/(n+0.5)
# of a bin's own probability: 33 percent at n=1, 4.8 percent at n=10, 0.5 percent at n=100.
# At n=1 the correction is dominated by the policy, so a bin below this floor is reported as
# thin. Judgement call, not a derived constant; the count and the reference mass involved are
# returned so a caller can re-gate without rerunning.
DEFAULT_MIN_BIN_OBS = 10

# How much of the reference probability may sit in bins the simulation never visited before
# the round is refused. Judgement call: at 0.02 the unmeasured region is 2 percent of the
# target, and the caller sees the measured number in every case.
DEFAULT_MAX_UNSUPPORTED_MASS = 0.02

# How many samples may fall outside [lo, hi] before the round is refused.
DEFAULT_MAX_OUTSIDE_FRAC = 0.01

# Ceiling on one round's applied correction, in kBT. The reference tables span 10-40 kJ/mol,
# so a single round moving more than about 50 kJ/mol anywhere is a sign the iteration is not
# in the regime the update assumes. Judgement call, and a parameter.
DEFAULT_MAX_STEP_KBT = 20.0

# Convergence tolerance on the applied correction, in kBT.
DEFAULT_TOL_KBT = 0.1

# Divergence rule: refuse when the applied correction has risen (not strictly) for PATIENCE
# consecutive rounds and is GROWTH times its value at the start of that window.
DEFAULT_PATIENCE = 3
DEFAULT_GROWTH = 2.0

TABLE_KEYS = ("lo", "hi", "binw", "U", "centre", "n", "empty")

REFUSE_NO_SAMPLES = "no_samples"
REFUSE_SIM_WHERE_REF_ZERO = "sim_mass_where_reference_is_zero"
REFUSE_SUPPORT_DRIFT = "support_drift"
REFUSE_STEP_TOO_LARGE = "step_too_large"
REFUSE_DIVERGENCE = "divergence"
REFUSE_UNSUPPORTED = "unvisited_reference_support"

STATUS_OK = "ok"
STATUS_CONVERGED = "converged"
STATUS_REFUSED = "refused"


class IBIRefusal(RuntimeError):
    """A round that produced no table. Carries the status, reason and diagnostics."""

    def __init__(self, status, reason, diagnostics):
        super().__init__(f"IBI round refused ({reason}): "
                         f"{diagnostics.get('message', reason)}")
        self.status = status
        self.reason = reason
        self.diagnostics = diagnostics


class SimHistogram(NamedTuple):
    """A sampled coordinate on the table's own bins.

    counts: per-bin counts on [lo, hi], boltzmann_bonded's binning convention.
    n: every observation offered, including the ones outside the support.
    n_outside: how many of those fell outside [lo, hi]. Counted, never silently dropped.
    lo/hi/nbins: the bins the counts were taken on, when the caller knows them. plan_update
    checks them against the table's own support when they are present, because "P_ref and
    P_sim on the same bins" is an assumption that must not be silent.
    """
    counts: np.ndarray
    n: int
    n_outside: int
    lo: float = None
    hi: float = None
    nbins: int = None


class UpdateResult:
    """One round: the next table, or why there is none.

    table is None whenever the round was refused. dU is the applied correction in kJ/mol, zero
    at every bin that was not measured, so the next table is the previous one plus dU. converged
    is a statement about the size of the correction, not about correctness; see the module
    docstring.
    """

    def __init__(self, *, table, dU, diagnostics, status, reason=None):
        self.table = table
        self.dU = dU
        self.diagnostics = diagnostics
        self.status = status
        self.reason = reason

    @property
    def ok(self):
        return self.table is not None

    @property
    def converged(self):
        return self.status == STATUS_CONVERGED

    @property
    def max_abs_dU(self):
        """The applied correction's largest magnitude, kJ/mol. Append to history."""
        return float(self.diagnostics["max_abs_dU"])

    def require_table(self):
        """The next table, or IBIRefusal. A caller that would use it anyway must call this."""
        if self.table is None:
            raise IBIRefusal(self.status, self.reason, self.diagnostics)
        return self.table

    def __repr__(self):
        return (f"UpdateResult(status={self.status!r}, reason={self.reason!r}, "
                f"max_abs_dU={self.max_abs_dU:.4g} kJ/mol)")


def probability_from_counts(counts, pseudo=DEFAULT_PSEUDO):
    """The pseudocount policy, in one place: (counts + pseudo) / (N + pseudo * nbins)."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError(f"counts must be 1-d, got shape {counts.shape}")
    if np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("counts must be finite and non-negative")
    if pseudo < 0 or not np.isfinite(pseudo):
        raise ValueError(f"pseudo must be a finite non-negative number, got {pseudo!r}")
    denom = counts.sum() + pseudo * counts.size
    if denom <= 0:
        raise ValueError("no observations and no pseudocount, so no probability")
    return (counts + pseudo) / denom


def probability_from_U(U, kBT=KBT):
    """The density a stored table implies, point weights: p ~ exp(-(U - min U)/kBT).

    This is what scripts/ibi_round0.py uses as its reference (its ref_p()). It treats the
    table as a set of point weights rather than as the piecewise-linear function that
    boltzmann_bonded._sample evaluates, and the two differ by kBT*ln[(1-e^-x)/x] per bin with
    x the free-energy step across that bin. Use bin_probabilities_from_U() for the
    convention-consistent target; this one is kept because it is the convention the existing
    round-0 measurement uses.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 1 or not np.all(np.isfinite(U)):
        raise ValueError("U must be a finite 1-d array")
    p = np.exp(-(U - U.min()) / kBT)
    return p / p.sum()


def probability_from_table(table, kBT=KBT):
    """probability_from_U(table["U"]), spelled for a table."""
    U, _, _, _, _, _ = _table_arrays(table)
    return probability_from_U(U, kBT=kBT)


def sample_segments(table):
    """The piecewise-linear function boltzmann_bonded._sample evaluates, as segments.

    _sample() interpolates between bin centres and clamps outside them, so the function the
    pipeline actually samples is flat at U[0] over [lo, centre[0]], linear between adjacent
    centres, and flat at U[-1] over [centre[-1], hi]. Those flat half-bins are part of the
    potential, not an artifact to drop: they are where a clamped table puts its edge mass,
    and leaving them out would make the target density of this module disagree with the
    density the sampler produces. Returns [(q0, q1, U0, U1), ...], tiling [lo, hi] with no
    overlap. The flat end segments carry U0 == U1.
    """
    U, lo, hi, binw, nbins, centre = _table_arrays(table)
    segs = []
    if centre[0] > lo:
        segs.append((lo, float(centre[0]), float(U[0]), float(U[0])))
    for j in range(nbins - 1):
        segs.append((float(centre[j]), float(centre[j + 1]), float(U[j]), float(U[j + 1])))
    if centre[-1] < hi:
        segs.append((float(centre[-1]), hi, float(U[-1]), float(U[-1])))
    return segs


def segment_moments(Ua, Ub, q0, length, a, b, kBT=KBT):
    """(I0, I1) over the part of one segment between fractions a and b of its length.

    The segment runs q = q0 + length * t for t in [0, 1] with U linear from Ua at t=0 to Ub
    at t=1, so U(q) = Ua + x * kBT * t and x = (Ub - Ua)/kBT. Then

        I0 = integral exp(-U/kBT) dq,   I1 = integral q exp(-U/kBT) dq,

    both analytic. The x -> 0 branch is the uniform-density value with its first correction,
    which avoids the 0/0 and the cancellation in the moment as the slope vanishes.
    """
    x = (Ub - Ua) / kBT
    E = np.exp(-Ua / kBT)
    if abs(x) < 1e-4:
        p0 = (b - a) * (1.0 - x * (a + b) / 2.0)
        p1 = (b ** 2 - a ** 2) / 2.0 - x * (b ** 3 - a ** 3) / 6.0
    else:
        exa, exb = np.exp(-x * a), np.exp(-x * b)
        p0 = (exa - exb) / x
        p1 = -((b / x + 1.0 / x ** 2) * exb - (a / x + 1.0 / x ** 2) * exa)
    I0 = length * E * p0
    I1 = q0 * I0 + length * length * E * p1
    return I0, I1


def segments_mean(segments, kBT=KBT):
    """The mean of q under exp(-U(q)/kBT) for a tiling segment list: I1 / I0."""
    total0 = 0.0
    total1 = 0.0
    for q0, q1, Ua, Ub in segments:
        i0, i1 = segment_moments(Ua, Ub, q0, q1 - q0, 0.0, 1.0, kBT=kBT)
        total0 += i0
        total1 += i1
    if total0 <= 0:
        raise ValueError("the segments carry no probability mass")
    return total1 / total0


def bin_probabilities_from_U(table, kBT=KBT):
    """Exact bin probabilities of the piecewise-linear table: the sampler's own convention.

    Each segment of sample_segments() is split at the bin edges it crosses and integrated
    analytically with segment_moments(), so no quadrature is involved and the answer is the
    bin integral of the same function boltzmann_bonded._sample evaluates. This is the target
    a sampler of the table reproduces, so it is the reference an update is checked against;
    probability_from_U() is the point-weight convention instead.
    """
    U, lo, hi, binw, nbins, centre = _table_arrays(table)
    edges = np.linspace(lo, hi, nbins + 1)
    p = np.zeros(nbins, dtype=float)
    for q0, q1, Ua, Ub in sample_segments(table):
        length = q1 - q0
        cuts = [q0] + [e for e in edges if q0 < e < q1] + [q1]
        for k in range(len(cuts) - 1):
            a, b = cuts[k], cuts[k + 1]
            i0, _ = segment_moments(Ua, Ub, q0, length, (a - q0) / length,
                                    (b - q0) / length, kBT=kBT)
            p[min(int((0.5 * (a + b) - lo) / binw), nbins - 1)] += i0
    total = float(p.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("the table's bin integrals do not normalise; U spans too much for "
                         "exp(-U/kBT) to be representable")
    return p / total


def load_clean_tables(npz_path, coords=B.COORDS):
    """The stored pooled tables, exactly the keys scripts/ibi_round0.py loads.

    Returns {coord: {lo, hi, binw, U, centre, sigma}}. No n/empty: the npz does not store
    them and inventing them would be a number nobody measured.
    """
    z = np.load(npz_path)
    return {name: {"lo": float(z[f"{name}__lo"]), "hi": float(z[f"{name}__hi"]),
                   "binw": float(z[f"{name}__binw"]),
                   "U": np.asarray(z[f"{name}__U"], dtype=float),
                   "centre": np.asarray(z[f"{name}__centre"], dtype=float),
                   "sigma": float(z[f"{name}__sigma"])}
            for name in coords}


def _table_arrays(table):
    """Validate a table and return (U, lo, hi, binw, nbins, centre)."""
    for key in ("lo", "hi", "binw", "U"):
        if key not in table:
            raise KeyError(f"table has no {key!r}; expected the boltzmann_bonded table "
                           f"format (lo, hi, binw, U, centre)")
    U = np.asarray(table["U"], dtype=float)
    if U.ndim != 1 or U.size < 3:
        raise ValueError(f"table U must be a 1-d array of at least 3 bins, got {U.shape}")
    if not np.all(np.isfinite(U)):
        raise ValueError("table U has non-finite entries")
    lo, hi = float(table["lo"]), float(table["hi"])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(f"table support is not a finite increasing interval: {lo}..{hi}")
    nbins = int(U.size)
    binw = float(table["binw"])
    expected = (hi - lo) / nbins
    if not np.isfinite(binw) or binw <= 0 or abs(binw - expected) > 1e-9 * binw:
        raise ValueError(f"table binw={binw!r} disagrees with (hi - lo)/nbins = {expected!r}")
    edges = np.linspace(lo, hi, nbins + 1)
    centre = 0.5 * (edges[:-1] + edges[1:])
    if "centre" in table:
        c = np.asarray(table["centre"], dtype=float)
        if c.shape != centre.shape or not np.allclose(c, centre, rtol=0.0, atol=1e-9 * binw):
            raise ValueError("table centre disagrees with lo/hi/nbins")
    return U, lo, hi, binw, nbins, centre


def bin_samples(values, table):
    """Count coordinate values on the table's own bins.

    Same convention as boltzmann_bonded._table_from_values: np.histogram(range=(lo, hi)), so
    a value equal to hi lands in the last bin and anything outside is reported as n_outside
    rather than dropped. scripts/ibi_round0.py bins by nearest bin centre instead
    (k = round((q - centre[0]) / binw)); that is the same assignment for every value except
    one exactly on a bin edge, where the two differ, and it drops a value equal to hi instead
    of counting it. Both cases are measure zero for a continuous coordinate, and they are
    named here rather than left as a silent mismatch.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(v)):
        raise ValueError("sampled values contain non-finite entries")
    _, lo, hi, _, nbins, _ = _table_arrays(table)
    counts, _ = np.histogram(v, bins=nbins, range=(lo, hi))
    n_outside = int(((v < lo) | (v > hi)).sum())
    return SimHistogram(counts=counts.astype(np.int64), n=int(v.size), n_outside=n_outside,
                        lo=lo, hi=hi, nbins=nbins)


def smooth_correction(dU, bins, mask=None):
    """Uniform moving average of a correction over +-bins bins, normalised over the mask.

    Only bins with mask True contribute, and the average is divided by the number of
    contributing bins, so a frozen (unmeasured) bin cannot leak a fabricated value into a
    measured neighbour. bins=0 returns a copy, bit-identical. Smoothing moves the fixed point;
    the diagnostics say when and how much it was applied.
    """
    dU = np.asarray(dU, dtype=float)
    if bins == 0:
        return dU.copy()
    if bins < 0:
        raise ValueError(f"smooth_bins must be non-negative, got {bins!r}")
    if 2 * bins + 1 > dU.size:
        raise ValueError(f"smooth_bins={bins} needs {2 * bins + 1} bins but the table has "
                         f"{dU.size}")
    if mask is None:
        mask = np.ones(dU.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != dU.shape:
        raise ValueError(f"mask shape {mask.shape} does not match dU shape {dU.shape}")
    out = dU.copy()
    n = dU.size
    for i in range(n):
        if not mask[i]:
            continue
        j0, j1 = max(0, i - bins), min(n, i + bins + 1)
        w = mask[j0:j1]
        out[i] = float(dU[j0:j1][w].mean())
    return out


def divergence_check(history, current, patience=DEFAULT_PATIENCE, growth=DEFAULT_GROWTH,
                     ceiling=None):
    """Does the correction look like a divergent iteration?

    history: previous rounds' applied max|dU| in kJ/mol, oldest first. current: this round's,
    before the update is applied. Rule: diverging if the correction has risen (not strictly)
    for 'patience' consecutive rounds and is 'growth' times its value at the start of that
    window, or if a ceiling was given and current exceeds it. A flat or falling correction is
    never called divergent, so this cannot fire on a run that is settling.

    Returns (diverging, message), the message carrying the measured numbers.
    """
    vals = [float(v) for v in history] + [float(current)]
    if any(not np.isfinite(v) or v < 0 for v in vals):
        raise ValueError(f"correction history must be finite and non-negative, got {vals}")
    if ceiling is not None and current > ceiling:
        return True, (f"this round's applied max|dU| = {current:.3f} kJ/mol exceeds the "
                      f"ceiling {ceiling:.3f} kJ/mol")
    if len(vals) >= patience + 1:
        window = vals[-(patience + 1):]
        rising = all(window[k + 1] >= window[k] for k in range(patience))
        if rising:
            if window[0] > 0:
                factor, grew = window[-1] / window[0], window[-1] >= growth * window[0]
            else:
                factor, grew = float("inf"), window[-1] > 0
            if grew:
                return True, (f"applied max|dU| rose for {patience} consecutive rounds, "
                              f"{window[0]:.4g} -> {window[-1]:.4g} kJ/mol (x{factor:.3g})")
    return False, None



def plan_update(table, hist, p_ref, pseudo=DEFAULT_PSEUDO, min_bin_obs=DEFAULT_MIN_BIN_OBS,
                max_unsupported_mass=DEFAULT_MAX_UNSUPPORTED_MASS,
                max_outside_frac=DEFAULT_MAX_OUTSIDE_FRAC, smooth_bins=0, gain=1.0,
                max_step_kbt=DEFAULT_MAX_STEP_KBT, tol_kbt=DEFAULT_TOL_KBT,
                history=(), patience=DEFAULT_PATIENCE, growth=DEFAULT_GROWTH):
    """One IBI round: the next table, or a refusal that says why there is none.

    table : the table that was simulated, boltzmann_bonded format (lo, hi, binw, U, centre).
    hist  : SimHistogram of the sampled coordinate on those same bins.
    p_ref : reference bin probabilities on those same bins; normalised here.

    Returns an UpdateResult whose .table carries exactly TABLE_KEYS, ready for
    boltzmann_bonded.prepare(). Its n and empty describe *this round's simulation*: n is the
    number of in-support observations, empty the number of bins it never visited.

    Checks, in the order they are reported. The diagnostics carry every measurement whether or
    not it fired, so a refused round is still readable:
    no_samples, sim_mass_where_reference_is_zero, support_drift, step_too_large, divergence,
    unvisited_reference_support.
    """
    U, lo, hi, binw, nbins, centre = _table_arrays(table)

    if not isinstance(hist, SimHistogram):
        raise TypeError(f"hist must be a SimHistogram from bin_samples(), got "
                        f"{type(hist).__name__}; pass the in-support counts and the number "
                        f"of samples that fell outside explicitly rather than letting "
                        f"either default")
    counts = np.asarray(hist.counts, dtype=float)
    if counts.shape != (nbins,):
        raise ValueError(f"histogram has {counts.shape} bins but the table has {nbins}")
    if hist.nbins is not None and int(hist.nbins) != nbins:
        raise ValueError(f"histogram was taken on {hist.nbins} bins, the table has {nbins}")
    if hist.lo is not None and abs(float(hist.lo) - lo) > 1e-12 * (hi - lo):
        raise ValueError(f"histogram support starts at {hist.lo}, the table at {lo}")
    if hist.hi is not None and abs(float(hist.hi) - hi) > 1e-12 * (hi - lo):
        raise ValueError(f"histogram support ends at {hist.hi}, the table at {hi}")
    if counts.sum() != hist.n - hist.n_outside:
        raise ValueError(f"histogram counts sum to {counts.sum()}, but n - n_outside = "
                         f"{hist.n} - {hist.n_outside}; the histogram is not self-consistent")

    p = np.asarray(p_ref, dtype=float)
    if p.shape != (nbins,):
        raise ValueError(f"p_ref has {p.shape}, the table has {nbins} bins; the reference "
                         f"and the simulation must be on the same bins")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("p_ref must be finite and non-negative")
    total = p.sum()
    if total <= 0:
        raise ValueError("p_ref sums to zero, so there is no target to iterate towards")
    p = p / total

    if not (0.0 <= max_unsupported_mass <= 1.0):
        raise ValueError(f"max_unsupported_mass must be a probability, got "
                         f"{max_unsupported_mass!r}")
    if not (0.0 <= max_outside_frac <= 1.0):
        raise ValueError(f"max_outside_frac must be a probability, got {max_outside_frac!r}")
    if min_bin_obs < 0:
        raise ValueError(f"min_bin_obs must be non-negative, got {min_bin_obs!r}")
    if smooth_bins != int(smooth_bins) or smooth_bins < 0:
        raise ValueError(f"smooth_bins must be a non-negative integer, got {smooth_bins!r}")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError(f"gain must be a finite positive number, got {gain!r}")
    if max_step_kbt <= 0 or not np.isfinite(max_step_kbt):
        raise ValueError(f"max_step_kbt must be finite and positive, got {max_step_kbt!r}")
    if tol_kbt < 0 or not np.isfinite(tol_kbt):
        raise ValueError(f"tol_kbt must be finite and non-negative, got {tol_kbt!r}")
    if int(patience) != patience or patience < 1:
        raise ValueError(f"patience must be a positive integer, got {patience!r}")
    if not np.isfinite(growth) or growth <= 0:
        raise ValueError(f"growth must be a finite positive number, got {growth!r}")
    # materialised once: a generator would be consumed by divergence_check() and then read as
    # an empty history by the diagnostics
    history = tuple(float(v) for v in history)

    n_sim = int(counts.sum())
    outside_frac = (hist.n_outside / hist.n) if hist.n else 0.0
    occupied = counts > 0
    ref_zero = p == 0
    unvisited = (~occupied) & (~ref_zero)      # reference has mass, simulation has none
    dead = (~occupied) & ref_zero              # neither has anything here
    sim_ref_zero = occupied & ref_zero         # simulation visits where the reference cannot
    measured = occupied & (~ref_zero)
    thin = measured & (counts < min_bin_obs)

    if counts.sum() + pseudo * nbins > 0:
        p_sim = probability_from_counts(counts, pseudo)
    else:                                      # the no_samples refusal fires before this is read
        p_sim = np.full(nbins, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ref_zero, np.nan, p_sim) / np.where(ref_zero, 1.0, p)
        dU_raw = KBT * np.log(ratio)
    dU_raw[dead] = 0.0
    dU_raw[unvisited] = 0.0

    applied = dU_raw.copy()
    applied[~measured] = 0.0
    if smooth_bins > 0:
        applied = smooth_correction(applied, int(smooth_bins), mask=measured)
    applied = gain * applied
    applied[~measured] = 0.0

    def max_abs(a):
        good = a[np.isfinite(a)]
        return float(np.abs(good).max()) if good.size else 0.0

    max_abs_raw = max_abs(dU_raw)
    max_abs_applied = max_abs(applied)
    ref_mass_unvisited = float(p[unvisited].sum())
    thin_ref_mass = float(p[thin].sum())
    sim_count_ref_zero = float(counts[sim_ref_zero].sum())

    refusal = None
    if n_sim == 0:
        refusal = (REFUSE_NO_SAMPLES,
                   f"the simulation contributed no observations inside [{lo:.6g}, {hi:.6g}]: "
                   f"{hist.n} offered, {hist.n_outside} of them outside, so there is no "
                   f"P_sim to compare with P_ref")
    elif sim_ref_zero.any():
        refusal = (REFUSE_SIM_WHERE_REF_ZERO,
                   f"the simulation put {sim_count_ref_zero:.0f} of {n_sim} observations "
                   f"({sim_count_ref_zero / n_sim:.4g} of them) in "
                   f"{int(sim_ref_zero.sum())} bin(s) where the reference probability is "
                   f"exactly zero; the correction there is infinite and the reference has "
                   f"no data to justify it")
    elif outside_frac > max_outside_frac:
        refusal = (REFUSE_SUPPORT_DRIFT,
                   f"{hist.n_outside} of {hist.n} observations "
                   f"({outside_frac:.4g}) fell outside the table support "
                   f"[{lo:.6g}, {hi:.6g}], above the {max_outside_frac:.4g} allowed; the "
                   f"histogram is a truncated view of the simulation, so the correction "
                   f"computed from it is biased. Widening the support needs reference data, "
                   f"not an update")
    elif max_abs_applied > max_step_kbt * KBT:
        refusal = (REFUSE_STEP_TOO_LARGE,
                   f"one round would move the table by {max_abs_applied:.3f} kJ/mol "
                   f"({max_abs_applied / KBT:.3f} kBT) somewhere, above the "
                   f"{max_step_kbt:.3f} kBT ceiling")
    else:
        diverging, div_message = divergence_check(history, max_abs_applied,
                                                  patience=patience, growth=growth)
        if diverging:
            refusal = (REFUSE_DIVERGENCE, div_message)
        elif ref_mass_unvisited > max_unsupported_mass:
            refusal = (REFUSE_UNSUPPORTED,
                       f"the simulation never visited {int(unvisited.sum())} of {nbins} "
                       f"bins holding {ref_mass_unvisited:.4g} of the reference probability, "
                       f"above the {max_unsupported_mass:.4g} allowed; the correction in "
                       f"those bins would be the pseudocount, not a measurement, and it "
                       f"would dominate the table")

    U_new = U + applied
    shift = float(U_new.min())
    U_new = U_new - shift
    converged = bool(refusal is None and not unvisited.any()
                     and max_abs_applied <= tol_kbt * KBT)

    if refusal is not None:
        status, reason, message = STATUS_REFUSED, refusal[0], refusal[1]
    else:
        status = STATUS_CONVERGED if converged else STATUS_OK
        reason, message = None, None

    new_table = None
    if refusal is None:
        new_table = {"lo": lo, "hi": hi, "binw": binw, "U": U_new, "centre": centre,
                     "n": n_sim, "empty": int((counts == 0).sum())}

    diagnostics = {
        "status": status, "reason": reason, "message": message,
        "converged": converged, "tol_kbt": float(tol_kbt), "tol": float(tol_kbt) * KBT,
        "nbins": nbins, "lo": lo, "hi": hi, "binw": binw,
        "n_sim": n_sim, "n_total": int(hist.n), "n_outside": int(hist.n_outside),
        "outside_frac": float(outside_frac), "max_outside_frac": float(max_outside_frac),
        "pseudo": float(pseudo), "min_bin_obs": int(min_bin_obs),
        "occupied_bins": int(occupied.sum()),
        "occupied_lo": int(np.argmax(occupied)) if occupied.any() else None,
        "occupied_hi": int(nbins - 1 - np.argmax(occupied[::-1])) if occupied.any() else None,
        "unvisited_bins": int(unvisited.sum()),
        "unvisited_ref_mass": ref_mass_unvisited,
        "max_unsupported_mass": float(max_unsupported_mass),
        "dead_bins": int(dead.sum()),
        "thin_bins": int(thin.sum()), "thin_counts": int(counts[thin].sum()),
        "thin_ref_mass": thin_ref_mass,
        "sim_bins_where_ref_zero": int(sim_ref_zero.sum()),
        "sim_count_where_ref_zero": sim_count_ref_zero,
        "sim_frac_where_ref_zero": (sim_count_ref_zero / n_sim) if n_sim else 0.0,
        "max_abs_dU_raw": max_abs_raw, "max_abs_dU": max_abs_applied,
        "rms_dU": float(np.sqrt(np.mean(applied[measured] ** 2))) if measured.any() else 0.0,
        "smooth_bins": int(smooth_bins), "smoothed": bool(smooth_bins > 0),
        "gain": float(gain), "shift": shift,
        "u_span": float(U_new.max() - U_new.min()),
        "history_len": len(history),
    }
    return UpdateResult(table=new_table, dU=applied, diagnostics=diagnostics, status=status,
                        reason=reason)


def advance_from_samples(table, values, p_ref, **policy):
    """bin_samples() then plan_update(), for a sampler that hands back raw coordinate values.

    p_ref is required and must be the round-0 reference. Deriving it from the table being
    updated would make the correction identically zero: see the module docstring.
    """
    return plan_update(table, bin_samples(values, table), p_ref, **policy)

