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
COORDS = ("bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack")
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

        lst = None
        for run in runs:
            if MIN_L < len(run) <= MAX_L:
                lst = run
                break
        if lst is None:
            continue
        # gly must be recomputed per residue; reusing the outer loop variable here silently
        # indexed every residue with the last one's base atom
        beads = np.array([[at["P"], at["C4'"],
                           at["N9" if r in ("A", "G") else "N1"]] for r, at in lst]) / 10.0
        pairs = []
        for a in range(len(lst)):   # noqa: E501  (unchanged below)
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
def _table_from_values(v, nbins=120, pseudo=0.5, support=None):
    """Bin one coordinate's observations and invert to U = -kBT ln P.

    support=None fits the range to v, which is what the pooled fit has always done.
    support=(lo, hi) bins on an existing range instead, so a subgroup table shares the
    pooled bin edges and can be compared with, or substituted for, the pooled table
    without a jump in either edge.
    """
    if support is None:
        lo, hi = float(v.min()), float(v.max())
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

# Support floor for a group table. At nbins=120 and pseudo=0.5 the pseudo-count carries
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


def fit(structs, nbins=120, pseudo=0.5, stratify=False, min_obs=MIN_OBS, coords=None):
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
        tables[name] = _table_from_values(v, nbins, pseudo)
    return tables


def stratify_values(v, lab, pooled, nbins=120, pseudo=0.5, min_obs=MIN_OBS):
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


def fit_stratified(structs, nbins=120, pseudo=0.5, min_obs=MIN_OBS, coords=None):
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
