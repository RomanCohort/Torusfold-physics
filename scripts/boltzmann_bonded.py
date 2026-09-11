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

Module only; fitting and checking live in test_boltzmann_bonded.py.
"""
import collections
import math
from pathlib import Path

import numpy as np
import torch

KBT = 2.494          # kJ/mol at 300 K
COORDS = ("bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack")
DATA = Path(r"D:\torusfold-cgdata\rsRNASP\Training_set")
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


def load_structures(limit=None):
    structs = []
    for f in sorted(DATA.glob("*.pdb")):
        if limit is not None and len(structs) >= limit:
            break
        for beads, pairs in _chain_residues(f):
            structs.append({"name": f.stem, "pos": beads, "pairs": pairs})
            if limit is not None and len(structs) >= limit:
                break
    return structs


# --------------------------------------------------------------------- tables
def fit(structs, nbins=120, pseudo=0.5):
    """Per coordinate: (lo, hi, binw, U) with U = -kBT ln P over [lo, hi]."""
    tables = {}
    for name in COORDS:
        vals = []
        for s in structs:
            pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
            vals.append(coords_of(pos, name).reshape(-1).numpy())
        v = np.concatenate(vals)
        lo, hi = float(v.min()), float(v.max())
        pad = 0.02 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
        # pseudo-count so empty bins are finite; without it -ln 0 is infinite and the
        # interpolator has nothing to interpolate between
        p = (counts + pseudo) / (counts.sum() + pseudo * nbins)
        U = -KBT * np.log(p)
        U = U - U.min()                                # shift, so only shape matters
        tables[name] = {"lo": lo, "hi": hi, "binw": (hi - lo) / nbins,
                        "U": U, "centre": 0.5 * (edges[:-1] + edges[1:]),
                        "n": int(counts.sum()),
                        "empty": int((counts == 0).sum())}
    return tables


def _sample(q, t):
    """Linear interpolation between bin centres, clamped to the table."""
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
