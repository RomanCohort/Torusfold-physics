"""Boltzmann-inverted bonded potentials for the 3-bead CG force field.

Replaces "target value + harmonic spring" with a tabulated free energy from the structure
database:  U(q) = -kBT ln P_ref(q),  following Direct Boltzmann Inversion as described in
Li & Chen, Biophys J 2026 (PMC13110076, Table 1).

Why. The harmonic bonded terms were not merely mistuned, they had the wrong functional form
for skewed coordinates. The P-P-P-P pseudo-torsion, for instance, has its mode at -22.5 deg
with only 5.1 percent of observations within 30 deg of 180; a harmonic spring to any single
value pins the structure somewhere most native configurations are not. A tabulated
U = -kBT ln P instead puts the minimum wherever the data puts the mode and prices the tails
by their actual frequency.

Forces are taken by autograd on U(q(x)) rather than hand-derived. torch_cgsim._dihedral_f
carries a comment admitting its gradient is an approximation ("The full formula is too
complex"), and tests/test_force_gradcheck.py fails on both paths at HEAD. Deriving a second
approximate gradient here would repeat that mistake, so the coordinate functions below are
plain differentiable torch and the gradient is exact by construction.

A -kBT ln P table is only defined on its support and interpolation flattens it outside, which
would leave the coordinate free to drift off the sampled range. Outside the fitted range a
C1-continuous quadratic wall is added, anchored on the edge value and the edge slope.

Stratification. fit() pools every observation of a coordinate into one histogram, so the
six tables are sequence averaged: base identity reaches the loader only to pick the N9-vs-N1
bead name and to decide which pairs count as Watson-Crick. Whether that averaging costs
anything is a measurement rather than an assumption, so fit() also has an opt-in stratified
form: fit_stratified() refits each coordinate once per base-identity group, on the pooled
support, and reports which groups had enough observations to be worth a table and which fell
back to the pooled one. The pooled path that every existing caller uses is not touched.

Module only; fitting and checking live in test_boltzmann_bonded.py.
"""
import collections
import math
import os
from pathlib import Path

import numpy as np
import torch

KBT = 2.494          # kJ/mol at 300 K
# THE BASE-LEVEL COORDINATES (2026-10-05) are the ones the TRACE cannot express: the base plane, its
# orientation relative to its neighbour's, and how far one base sits above the other along the mean
# normal. They are functions of the three beads alone, through the rigid template map
# (torusfold.scheme2.base_frames, one pooled triple, 6.573 degrees off the template's own planes), so a
# potential can use them and so can this file.
#
# They are SCORED, never injected: the dynamics that moves them is base_stacking's term, and the tables
# here are the crystal-marginal targets the loop measures against -- the same role `stack` has had since
# it was found to be an algebraic function of the bond and the angle.
COORDS = ("bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack")

# THE BASE-LEVEL COORDINATES (2026-10-05) are the ones the TRACE cannot express: the base plane, its
# orientation relative to its neighbour's, and how far one base sits above the other along the mean
# normal. They are functions of the three beads alone through the rigid template map
# (torusfold.scheme2.base_stacking.base_normals), so a potential can use them and so can this file.
#
# They are kept OUT of COORDS, and that is deliberate rather than tidy: COORDS is what a table file must
# contain for every existing caller, and adding names to it makes results/refit_smooth5.npz unloadable
# (measured: load_tables raises "missing 18 key(s)" the moment the tuple grows). scored_coords() below is
# the opt-in way in: with IBI_SCORE_BASE=1 a round bins and scores them as well, and every table file then
# has to carry them (scripts/build_base_level_ref.py writes the merged reference).
BASE_COORDS = ("base_dist", "base_rise", "base_cos")
_scored = None


def scored_coords():
    """The coordinates a round bins: COORDS, plus BASE_COORDS when IBI_SCORE_BASE=1.

    Cached, because ibi_core calls it in its per-frame loops and the environment does not change mid-run.
    Default (env unset) returns COORDS itself, so every shipped path is bit-identical.
    """
    global _scored
    if _scored is None:
        _scored = COORDS + (BASE_COORDS if os.environ.get("IBI_SCORE_BASE", "0") == "1" else ())
    return _scored
# The two coordinates the model holds RIGID, by SHAKE/RATTLE, rather than by a spring
# (src/torusfold/scheme2/rigid_bonds.py; the field's K_INTRA_PC/K_INTRA_CN are deleted).
#
# They belong to COORDS -- the reference tables for them are real, measured from deposited
# structures -- but they do NOT belong in an IBI round or in the residual that scores one:
#
#   * their simulated distribution is a delta at the constraint target, not the field's
#     equilibrium distribution, so kBT*ln(P_sim/P_ref) is meaningless for them;
#   * sigma_sim is EXACTLY ZERO, measured over 48 observations of a six-residue round: SHAKE
#     puts the distance in the same place to the last bit every frame, so the per-frame spread
#     is 0.0, not merely small.
#
# The second point does NOT do what it looks like it should. A zero ratio would give
# |ln(sim/ref)| = inf, and one such term would swamp the mean -- which is what this comment
# claimed until it was measured. simref's `r > 0` guard counts a zero ratio out instead, so the
# coordinate is DROPPED from the average and the denominator silently shrinks. A constrained
# round scored without an explicit skip would report a J averaged over four coordinates while
# the run's configuration says six, and nothing in the output would say so: the number would look
# like a plain improvement.
#
# That is worse than a bad number, because a bad number gets investigated. Hence two defences:
# run_round skips these by default, AND it prints the denominator next to J (res.j_coords), so a
# shrinking J is visible as "4/6" rather than as progress.
#
# ibi_update additionally refuses to write a table for them.
CONSTRAINED = ("intra_pc", "intra_cn")

# The two coordinates whose domain is [-1, 1] rather than the whole line. They are the reason
# _table_from_values takes a `bounded` flag: everything else gets a sigma window, these get
# [min, max] because their wall must sit outside the physical domain. See that function.
COSINE_COORDS = ("angle", "dihedral")

# Histogram resolution. 120 was the default until 2026-09-14; on the sigma-window support above
# it left the peak covered by only ~30 bins for bb_bond (sigma 0.0539 nm over a 0.42 nm window
# is ~8 bins per sigma, and the distribution's core is narrow), which is coarse for a potential
# that IBI then differentiates. 1000 bins costs 8 KB per coordinate in the npz.
DEFAULT_NBINS = 1000

# Half-width of the sigma window the support is built from, in robust sigmas. 4 reproduces the
# table of record's bb_bond support to 0.3 percent; see _table_from_values.
SUPPORT_SIGMA = 4.0
# The deposited-structure database. It lives OUTSIDE this repository (191 PDB files, 686.7 MB), so
# a checkout on another machine has to be told where it is. TORUSFOLD_RSRNASP overrides; the
# Windows default is kept so nothing here changes on the machine it was measured on.
#
# Only THIS file matters for the IBI chain: ibi_round0.py, ibi_bonded.py and
# sample_bonded_chain.py all reach the database through boltzmann_bonded. Seventeen other scripts
# carry their own copy of the same literal and need editing individually if they are to run
# elsewhere; see docs/dev_machine_handoff.md.
import _cgdata
DATA = _cgdata.rsrnasp()
MIN_L, MAX_L = 20, 120


# ---------------------------------------------------------------- coordinates
def _norm(v, dim=-1, eps=1e-6):
    return torch.clamp(torch.linalg.norm(v, dim=dim, keepdim=True), min=eps)


def _cos_angle(a, b, c):
    v1, v2 = a - b, c - b
    return (v1 * v2).sum(-1) / (_norm(v1).squeeze(-1) * _norm(v2).squeeze(-1))


def _cos_dihedral(p0, p1, p2, p3):
    """Same normal-based cosine the force field uses, so the fitted distribution matches."""
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0 = torch.linalg.cross(b0, b1, dim=-1)
    n1 = torch.linalg.cross(b1, b2, dim=-1)
    return (n0 * n1).sum(-1) / (_norm(n0).squeeze(-1) * _norm(n1).squeeze(-1))


def coords_of(pos, name):
    """pos: (B, 3L, 3) nm. Returns (B, M) for the named coordinate."""
    B, N, _ = pos.shape
    L = N // 3
    P = lambda i: 3 * i
    C4 = lambda i: 3 * i + 1
    NN = lambda i: 3 * i + 2
    dev = pos.device
    idx = torch.arange(L, device=dev)

    if name == "bb_bond":
        return torch.linalg.norm(pos[:, P(idx[:-1])] - pos[:, P(idx[1:])], dim=-1)
    if name == "intra_pc":
        return torch.linalg.norm(pos[:, P(idx)] - pos[:, C4(idx)], dim=-1)
    if name == "intra_cn":
        return torch.linalg.norm(pos[:, C4(idx)] - pos[:, NN(idx)], dim=-1)
    if name == "stack":
        st = torch.arange(L - 2, device=dev)
        return torch.linalg.norm(pos[:, P(st)] - pos[:, P(st + 2)], dim=-1)
    if name == "angle":
        a = torch.arange(L - 2, device=dev)
        return _cos_angle(pos[:, P(a)], pos[:, P(a + 1)], pos[:, P(a + 2)])
    if name == "dihedral":
        a = torch.arange(L - 3, device=dev)
        return _cos_dihedral(pos[:, P(a)], pos[:, P(a + 1)],
                             pos[:, P(a + 2)], pos[:, P(a + 3)])
    if name in ("base_dist", "base_rise", "base_cos"):
        # Imported here rather than at module scope: base_stacking lives in src/torusfold/scheme2 and
        # this module is imported by the loop's workers before any torch device is chosen.
        # base_normals defaults to the nm-scaled coefficient triple, which is the unit pos comes in.
        from torusfold.scheme2.base_stacking import base_normals  # noqa: PLC0415
        if L < 2:
            return torch.zeros(pos.shape[0], 0, dtype=pos.dtype, device=dev)
        n = base_normals(pos)                                   # (B, L, 3) unit base-plane normals
        nb = pos[:, NN(idx)]                                    # (B, L, 3) the N9/N1 beads
        ni, nj = n[:, :-1, :], n[:, 1:, :]
        nb_i, nb_j = nb[:, :-1, :], nb[:, 1:, :]
        # SIGN-ALIGNED, because the map's own convention fixes each normal's sign but a stacked pair's
        # two normals can still come out opposed when the two residues' rigid frames are rotated
        # relative to one another; the angle we want is the folded one.
        flip = (ni * nj).sum(-1, keepdim=True) < 0
        nj = torch.where(flip, -nj, nj)
        nm = ni + nj
        nrm = torch.linalg.norm(nm, dim=-1, keepdim=True)
        nm = torch.where(nrm > 1e-9, nm / nrm.clamp_min(1e-12), ni)
        dc = nb_j - nb_i
        if name == "base_dist":
            return torch.linalg.norm(dc, dim=-1)
        if name == "base_rise":
            return (dc * nm).sum(-1)
        return (ni * nj).sum(-1).clamp(-1.0, 1.0)
    raise KeyError(name)


# -------------------------------------------------------------------- fitting
WCP = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def _chain_residues(pdb, with_names=False):
    """(beads, wc_pairs) per chain. C1' is kept because the pair criterion needs it.

    with_names=True appends the per-residue base letter, so a caller can ask whether a
    fitted coordinate depends on base identity. The tables themselves never see it.
    """
    ch = collections.OrderedDict()
    for line in open(pdb):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        rname = line[17:20].strip()
        if rname not in ("A", "C", "G", "U"):
            continue
        aname = line[12:16].strip()
        gly = "N9" if rname in ("A", "G") else "N1"
        if aname not in ("P", "C4'", gly, "C1'"):
            continue
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        rec = ch.setdefault(line[21], collections.OrderedDict())
        rec.setdefault((line[22:27].strip(), rname), {})[aname] = xyz
    out = []
    for rec in ch.values():
        kept = []
        for (rid, rname), at in rec.items():
            gly = "N9" if rname in ("A", "G") else "N1"
            if not all(k in at for k in ("P", "C4'", gly, "C1'")):
                continue
            kept.append((rid, rname, at))

        # Split at numbering gaps. Keeping only A/C/G/U silently drops modified
        # nucleotides, so two entries that are adjacent in this list can be residues
        # several apart in the chain. Measured over rsRNASP/Training_set, 93 of 7354
        # consecutive pairs were such gaps, with steps of 2 to 10. A gap pair would
        # otherwise be counted as a backbone bond, an angle and three dihedrals that do
        # not exist, contaminating exactly the distributions the tables are fitted to.
        runs, cur = [], None
        for rid, rname, at in kept:
            try:
                n = int(rid)
            except ValueError:
                n = None
            if cur is not None and n is not None and prev is not None and n == prev + 1:
                cur.append((rname, at))
            else:
                if cur:
                    runs.append(cur)
                cur = [(rname, at)]
            prev = n
        if cur:
            runs.append(cur)

        # Every run, at its FULL length. There were two rejections here and both are gone:
        # an upper cap (MIN_L < len <= MAX_L, i.e. 20..120) and a `break` that kept only the
        # FIRST qualifying run per chain.
        #
        # What the two did together, measured on _cgdata/combined (439 files, 2026-09-14):
        # 166 files yielded NOTHING. A clean 3000-residue chain is a single run, and
        # `3000 <= 120` is false -- it was REJECTED, not truncated. The 294 chains that
        # survived were the broken-up ones, i.e. fragments, topping out at exactly 120. So the
        # "expanded" database contributed fragments and no length coverage at all, which is why
        # the 439 fit's bb_bond sigma moved (0.0471 -> 0.0531) while its chain lengths did not.
        #
        # MAX_L stays defined because check_loader_ordering.py and measure_pair_weight_quality.py
        # read B.MAX_L; the loader just no longer uses it. The one rejection left is a run too
        # short to carry a distribution -- a run of <= MIN_L cannot support a 120-bin histogram,
        # and that is a statement about statistics, not about length.
        #
        # THE WEIGHTING CONSEQUENCE, stated because it is large and not obvious: the pooled
        # histogram is now dominated by the longest chains. A 3611-residue ribosome contributes
        # 3611 observations of every local coordinate against roughly 50 from a small fragment.
        # That is the point of removing the cap, and it is also the caveat -- these long chains
        # are protein-held cryo-EM complexes, not free RNA.
        for lst in runs:
            if len(lst) <= MIN_L:
                continue
            # gly must be recomputed per residue; reusing the outer loop variable here silently
            # indexed every residue with the last one's base atom
            beads = np.array([[at["P"], at["C4'"],
                               at["N9" if r in ("A", "G") else "N1"]] for r, at in lst]) / 10.0
            pairs = []
            for a in range(len(lst)):
                for b in range(a + 3, len(lst)):
                    if (lst[a][0], lst[b][0]) not in WCP:
                        continue
                    d = np.linalg.norm(np.array(lst[a][1]["C1'"]) - np.array(lst[b][1]["C1'"]))
                    if 9.0 <= d <= 11.5:
                        pairs.append((a, b))
            out.append((beads, pairs, [r for r, _ in lst]) if with_names
                       else (beads, pairs))
    return out


def load_structures(limit=None, with_names=False):
    """Chain records. with_names=True adds "names", the per-residue base letters.

    The default record is byte-for-byte what every existing caller already gets; the
    letters are opt-in because fit() ignores them and only the stratified fit needs them.
    """
    structs = []
    for f in sorted(DATA.glob("*.pdb")):
        if limit is not None and len(structs) >= limit:
            break
        for rec in _chain_residues(f, with_names=with_names):
            if with_names:
                beads, pairs, names = rec
                s = {"name": f.stem, "pos": beads, "pairs": pairs, "names": names}
            else:
                beads, pairs = rec
                s = {"name": f.stem, "pos": beads, "pairs": pairs}
            structs.append(s)
            if limit is not None and len(structs) >= limit:
                break
    return structs


# --------------------------------------------------------------------- tables
# Moving-average width, in BINS, applied to a fitted table's U before it is used as a potential.
# NOT the same convention as ibi_bonded.smooth_correction(dU, bins), which averages over +-bins;
# the conversion is bins = (SMOOTH_WIDTH - 1) // 2 and getting it wrong smooths one path twice as
# hard as the other.
#
# WHY A TABLE HAS TO BE SMOOTHED AT ALL, measured 2026-09-14 on 1L2X, 16 replicas, 40 ps, the
# fitted table injected, friction 1.0, kinetic temperature after eight 5 ps blocks:
#
#     table                 T settles at    bb_bond outside support    bb_bond sd
#     none (shipped field)     313 K               0.00%                 0.057
#     fitted, unsmoothed       588 K               1.68%                 0.080
#     fitted, width 5          301 K               0.00%                 0.060
#     fitted, width 21         296 K               0.00%                 0.058
#
# U = -kBT ln p is a log of a histogram, so it carries Poisson noise of order kBT*sqrt(1/n) --
# about 0.06-0.1 kBT per bin at a few hundred counts. The interpolant is piecewise linear, so that
# noise IS a random force field, and a symplectic integrator pumps energy on a discontinuous
# force. At width 1 the mean |dU| between neighbouring bins is 0.588 kJ/mol over a 0.000436 nm
# bin, i.e. force jumps of 1349 kJ/mol/nm -- larger than K_BB itself. Smoothing removes the noise
# and the heating with it: the injected system then runs at the 300 K target and bb_bond's
# sampled spread comes out 0.060 against a reference of 0.0633, sim/ref = 0.95.
SMOOTH_WIDTH = 5


def smooth_U(U, width=SMOOTH_WIDTH):
    """Uniform moving average of a table's U over `width` bins, edges padded.

    width=1 returns a copy, bit-identical, so a caller can turn this off without a branch.
    """
    U = np.asarray(U, dtype=float)
    if width <= 1:
        return U.copy()
    w = int(width)
    pad = w // 2
    k = np.ones(w) / w
    return np.convolve(np.pad(U, pad, mode="edge"), k, mode="valid")[:len(U)]


def _table_from_values(v, nbins=DEFAULT_NBINS, pseudo=0.5, support=None, bounded=False):
    """Bin one coordinate's observations and invert to U = -kBT ln P.

    support=None fits the range to v, which is what the pooled fit has always done.
    support=(lo, hi) bins on an existing range instead, so a subgroup table shares the
    pooled bin edges and can be compared with, or substituted for, the pooled table
    without a jump in either edge.

    support=None now means a SIGMA WINDOW, not [min, max] plus a pad, for every unbounded
    coordinate. Why it had to change: on the expanded 867-chain database bb_bond runs
    0.3161 .. 7.2240 nm, and 44 observations out of 132,695 are above 0.8 nm. A P-P bond of
    7 nm does not exist -- those are the few adjacent-numbered pairs that a cryo-EM ribosome
    still has after the numbering-gap split. Taking [min, max] let those 44 set the axis, so
    the table spanned [0.178, 7.362] and 88 of 120 bins were empty: the whole distribution
    lived in three or four bins. Percentiles do not fix it either -- the tail is sparse but
    long, so clipping 1e-4 still leaves the support at about 1.5 nm and clipping 1e-5 leaves
    it at 7.

    So the support is the median plus and minus SUPPORT_SIGMA robust sigmas, with the sigma
    taken from the [0.1, 99.9] percentile span (6.58 sigma for a normal). At SUPPORT_SIGMA=4
    this reproduces the table of record's own bb_bond support almost exactly -- 0.382..0.801
    against its 0.3837..0.7958 -- which is the check that the window is not an invention.

    bounded=True keeps [min, max] and is for the two COSINE coordinates. Their domain is
    [-1, 1] and the wall that table_wall puts outside the support is what stops the sampler
    leaving it, so shrinking the support to a sigma window would move that wall INSIDE the
    coordinate's physical domain. The data has values at exactly -1.0000, so their [min, max]
    is already the domain, and it is the one case where the extremes are the right answer.
    """
    if support is None:
        if bounded:
            lo, hi = float(v.min()), float(v.max())
        else:
            med = float(np.median(v))
            q_lo, q_hi = np.quantile(v, [0.001, 0.999])
            sigma = float(q_hi - q_lo) / 6.58
            half = SUPPORT_SIGMA * sigma
            lo, hi = med - half, med + half
            # never invent range the data does not have: a narrow coordinate keeps its own edges
            lo, hi = max(lo, float(v.min())), min(hi, float(v.max()))
        pad = 0.02 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = float(support[0]), float(support[1])
    counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    # pseudo-count so empty bins are finite; without it -ln 0 is infinite and the
    # interpolator has nothing to interpolate between
    p = (counts + pseudo) / (counts.sum() + pseudo * nbins)
    U = -KBT * np.log(p)
    U = U - U.min()                                # shift, so only shape matters
    return {"lo": lo, "hi": hi, "binw": (hi - lo) / nbins,
            "U": U, "centre": 0.5 * (edges[:-1] + edges[1:]),
            "n": int(counts.sum()),
            "empty": int((counts == 0).sum())}


# ---------------------------------------------------------------- stratification
# How many residues each coordinate spans, endpoints included. coords_of() emits
# (L - SPAN + 1) windows per chain, so a label built from SPAN consecutive residues lines
# up one-for-one with that output. labels_for() checks the counts against coords_of()
# rather than trusting this table, because an off-by-one here would mis-assign every
# observation to the wrong base and nothing downstream would raise.
SPAN = {"bb_bond": 2, "intra_pc": 1, "intra_cn": 1,
        "angle": 3, "dihedral": 4, "stack": 3}

# A scheme is the set of window-relative offsets whose base letters are concatenated into
# the group label. 5-prime to 3-prime order is kept and never sorted: these coordinates
# are directional, and a UA step is not an AU step.
SCHEMES = {
    "residue":      (0,),        # the one residue the coordinate sits in
    "pair":         (0, 1),      # the two residues the coordinate joins
    "skip":         (0, 2),      # the two residues two apart
    "triplet":      (0, 1, 2),
    "middle":       (1,),        # the residue between the endpoints
    "central_pair": (1, 2),      # the two residues the central bond joins
}

# Which labelling each coordinate is stratified by, and why:
#   intra_pc / intra_cn  P-C4' and C4'-N both live inside one residue, so the base at that
#                        residue is the only candidate.
#   bb_bond              spans two residues; the ordered pair across the bond is the
#                        minimal complete label.
#   stack                spans i and i+2; same argument, with a one-residue gap.
#   dihedral             a P-P-P-P pseudo-torsion turns about the i+1/i+2 bond, so those
#                        two bases are the ones whose orientation the coordinate reports.
#   angle                spans three residues. The full triplet (64 groups) was measured
#                        to be unsupportable at this database size -- its largest group is
#                        smaller than the support floor -- so the middle residue, which is
#                        the one the two P-P vectors share, is used.
DEFAULT_SCHEME = {
    "bb_bond":  "pair",
    "intra_pc": "residue",
    "intra_cn": "residue",
    "angle":    "middle",
    "dihedral": "central_pair",
    "stack":    "skip",
}

# Support floor for a group table. At nbins=DEFAULT_NBINS and pseudo=0.5 the pseudo-count carries
# 0.5*120/(n + 0.5*120) of the probability mass: 23 percent at n=200, 11 percent at 500.
# 200 is chosen to be permissive -- it is the smallest floor at which every pair-labelled
# scheme stays fully supported on the whole 191-file database, so stratification is
# actually exercised instead of silently collapsing to the pooled table. It is a judgement
# call, not a derived constant; the per-group counts and pseudo fractions are returned so
# a caller can re-gate without refitting, and refit_tables_stratified.py sweeps it.
MIN_OBS = 200


def labels_for(names, coord, scheme=None):
    """Base-identity label for every observation coords_of(..., coord) emits, in order.

    names is the per-residue letter list from _chain_residues(..., with_names=True). The
    returned array is positional: element k labels window k of the coordinate array.
    """
    sch = DEFAULT_SCHEME[coord] if scheme is None else scheme
    if sch not in SCHEMES:
        raise KeyError(f"unknown stratification scheme {sch!r}; have {sorted(SCHEMES)}")
    off = SCHEMES[sch]
    if off[-1] >= SPAN[coord]:
        raise ValueError(f"scheme {sch!r} reaches residue offset {off[-1]}, but {coord} "
                         f"spans only {SPAN[coord]} residue(s)")
    n = np.asarray(names)
    if n.ndim != 1:
        # a bare string would become a 0-d array here and every later len() would raise
        raise ValueError(f"names must be a 1-d sequence of per-residue letters, got shape "
                         f"{n.shape}; pass list(sequence) if it is a string")
    win = len(n) - SPAN[coord] + 1
    if win <= 0:
        return np.array([], dtype="<U1")
    parts = [n[o:o + win] for o in off]
    return np.array(["".join(row) for row in zip(*parts)])


def fit(structs, nbins=DEFAULT_NBINS, pseudo=0.5, stratify=False, min_obs=MIN_OBS, coords=None):
    """Per coordinate: (lo, hi, binw, U) with U = -kBT ln P over [lo, hi].

    stratify=False (the default, and what every existing caller gets) returns exactly the
    pooled tables described above. stratify=True returns fit_stratified()'s nested
    structure instead -- a deliberately different shape behind an explicit flag, so that no
    call site can change meaning by accident.
    """
    if stratify:
        return fit_stratified(structs, nbins=nbins, pseudo=pseudo,
                              min_obs=min_obs, coords=coords)
    tables = {}
    for name in COORDS:
        vals = []
        for s in structs:
            pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
            vals.append(coords_of(pos, name).reshape(-1).numpy())
        v = np.concatenate(vals)
        tables[name] = _table_from_values(v, nbins, pseudo, bounded=(name in COSINE_COORDS))
    return tables


def stratify_values(v, lab, pooled, nbins=DEFAULT_NBINS, pseudo=0.5, min_obs=MIN_OBS):
    """Group tables for one coordinate's observations, on the pooled support.

    Split out of fit_stratified() so the null control in refit_tables_stratified.py runs
    the same code the builder runs -- a measurement that reimplements the thing it is
    measuring is measuring the reimplementation.

    Returns (groups, counts, fallback). Groups below min_obs are absent from groups and
    named in fallback; their observed counts are still in counts, so a reader can see how
    far short each one fell instead of having to guess.
    """
    groups, counts, fallback = {}, {}, []
    for g in np.unique(lab):
        m = lab == g
        counts[str(g)] = int(m.sum())
        if counts[str(g)] < min_obs:
            fallback.append(str(g))
            continue
        vg = v[m]
        t = _table_from_values(vg, nbins, pseudo, support=(pooled["lo"], pooled["hi"]))
        # kBT/sigma^2 is the harmonic stiffness that would reproduce this group's spread,
        # which is the number the local terms already use as their criterion.
        t["sigma"] = float(vg.std())
        t["k"] = KBT / t["sigma"] ** 2
        t["pseudo_frac"] = pseudo * nbins / (counts[str(g)] + pseudo * nbins)
        groups[str(g)] = t
    return groups, counts, fallback


def fit_stratified(structs, nbins=DEFAULT_NBINS, pseudo=0.5, min_obs=MIN_OBS, coords=None):
    """One table per base-identity group, alongside the pooled table it would replace.

    structs must come from load_structures(..., with_names=True).

    Every group table is binned on the pooled support: same [lo, hi], same bin width. A
    group table and the pooled table are therefore directly comparable, and a caller may
    substitute one for the other without introducing a discontinuity at the edges.

    Returns {coord: {"pooled", "groups", "n", "fallback", "min_obs", "labelling",
    "n_obs", "n_groups", "pooled_sigma", "pooled_k"}} where "n" carries the observed
    count of EVERY group, supported or not, and "fallback" names the ones priced by the
    pooled table. The fallback is reported, never silent.
    """
    if any("names" not in s for s in structs):
        raise ValueError(
            "fit_stratified needs records from load_structures(..., with_names=True); "
            "these carry no base letters, so there is nothing to stratify by")
    coords = tuple(COORDS if coords is None else coords)
    pooled = fit(structs, nbins=nbins, pseudo=pseudo)
    vals = {c: [] for c in coords}
    labs = {c: [] for c in coords}
    for s in structs:
        pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
        for c in coords:
            v = coords_of(pos, c).reshape(-1).numpy()
            lab = labels_for(s["names"], c)
            if len(lab) != len(v):
                raise ValueError(f"{c}: {len(lab)} base labels for {len(v)} observations; "
                                 f"the labelling does not line up with the coordinate")
            vals[c].append(v)
            labs[c].append(lab)
    out = {}
    for c in coords:
        v = np.concatenate(vals[c])
        lab = np.concatenate(labs[c])
        groups, counts, fallback = stratify_values(
            v, lab, pooled[c], nbins=nbins, pseudo=pseudo, min_obs=min_obs)
        out[c] = {"pooled": pooled[c], "groups": groups, "n": counts,
                  "fallback": sorted(fallback), "min_obs": min_obs,
                  "labelling": DEFAULT_SCHEME[c], "n_obs": int(len(v)),
                  "n_groups": int(len(counts)),
                  "pooled_sigma": float(v.std()), "pooled_k": KBT / float(v.std()) ** 2}
    return out


def _sample(q, t):
    """Linear interpolation between bin centres, clamped to the table.

    READ THIS BEFORE CHANGING THE INTERPOLATION. t["U"] holds U_i = -kBT ln p_i where p_i is the
    probability MASS of bin i, so the stored numbers are bin AVERAGES. This function reads them as
    point values at the bin centres. Those are different quantities, and the difference is
    measurable: with this interpolant the bin masses it produces are off by up to

        angle     0.3624 kBT at the core       mass-weighted rms 0.034 kBT
        dihedral  0.4925 kBT                   mass-weighted rms 0.147 kBT
        stack     0.2848 kBT                   mass-weighted rms 0.027 kBT

    over 126 chains, 120 bins each, core = bins whose target mass is at least 1e-4
    (scripts/fix_table_interpolation_convention.py prints the table). The all-bin maximum is owned
    by the pseudo-count and is not the number to quote.

    The gap is NOT closed. Inverting "node values -> bin masses" is ill-posed here: restricted to
    the populated bins the Jacobian has condition number ~4e7, the Newton step it implies is 1e4 to
    1e5 kJ/mol, and a 0.02x damped step along that direction does not reduce the residual. The
    whole attempt is in that script.

    It does not need closing. IBI is self-correcting for any convention, because its fixed point --
    the SIMULATED bin masses equal the target -- is stated in terms of the simulation and not of
    the interpolant; and the shipped constants come from the spread (k = kBT/sigma^2), not from the
    table shape. DBI's literal output is the one place the gap survives, and it is recorded there.
    """
    u = (q - t["lo"]) / t["binw"] - 0.5
    i0 = torch.floor(u).long().clamp(0, len(t["U"]) - 2)
    f = (u - i0).clamp(0.0, 1.0)
    U = t["Ut"]
    return U[i0] + f * (U[i0 + 1] - U[i0])


def prepare(tables, device="cpu"):
    """Cache torch tensors, the edge slopes of the wall, and the harmonic target."""
    for t in tables.values():
        U = torch.tensor(t["U"], dtype=torch.float64, device=device)
        t["Ut"] = U
        t["slope_lo"] = float(max((U[1] - U[0]) / t["binw"], 0.0))
        t["slope_hi"] = float(max((U[-1] - U[-2]) / t["binw"], 0.0))
        # the harmonic fallback needs a target; the table's own minimum is the mode, which
        # is the same criterion used everywhere else in this file
        t["target"] = float(t["centre"][int(np.argmin(t["U"]))])
        if "sigma" not in t:
            t["sigma"] = float(t["U"].size and 0.0)
    return tables


TABLED = ("angle", "dihedral", "stack")
HARMONIC = ("bb_bond", "intra_pc", "intra_cn")


def mixed_energy(pos, tables, k_local=None, k_wall=200.0, which=None):
    """Harmonic local terms plus tabulated angular ones.

    The split follows the measurement in refit_tables_clean.py. The three local coordinates
    have narrow, near-Gaussian distributions, and their shipped constants are 2 to 91 times
    too soft, so setting k = kBT/sigma^2 reproduces the observed spread exactly and no table
    can improve on that. The angle, dihedral and stack distributions are broad and skewed --
    the dihedral spans most of the cosine range -- and their shipped constants are 21 and 69
    times too stiff, so they get the tables.

    k_local: {name: k}. Defaults to kBT/sigma^2 from the tables' own stored sigma.
    which: iterable of coordinate names to include; defaults to all six.
    """
    if which is None:
        which = COORDS
    total = None
    for name in which:
        t = tables[name]
        if name in HARMONIC:
            k = (k_local or {}).get(name, KBT / t["sigma"] ** 2)
            q = coords_of(pos, name)
            e = (0.5 * k * (q - t["target"]) ** 2).sum()
        else:
            q = coords_of(pos, name)
            e = _sample(q, t)
            d_lo = (t["lo"] - q).clamp(min=0.0)
            d_hi = (q - t["hi"]).clamp(min=0.0)
            e = e + t["slope_lo"] * d_lo + 0.5 * k_wall * d_lo ** 2
            e = e + t["slope_hi"] * d_hi + 0.5 * k_wall * d_hi ** 2
            e = e.sum()
        total = e if total is None else total + e
    return total


def energy(pos, tables, k_wall=200.0):
    """Sum of tabulated bonded potentials, in kJ/mol. Differentiable w.r.t. pos."""
    total = None
    for name in COORDS:
        t = tables[name]
        q = coords_of(pos, name)
        e = _sample(q, t)
        d_lo = (t["lo"] - q).clamp(min=0.0)
        d_hi = (q - t["hi"]).clamp(min=0.0)
        e = e + t["slope_lo"] * d_lo + 0.5 * k_wall * d_lo ** 2
        e = e + t["slope_hi"] * d_hi + 0.5 * k_wall * d_hi ** 2
        total = e.sum() if total is None else total + e.sum()
    return total
