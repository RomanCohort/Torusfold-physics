"""A differentiable evaluator for the cgRNASP coarse-grained statistical potential.

Scheme and tables from Tan-group/cgRNASP (NAR Genom Bioinform 2023;5(1):lqad016), read
from their C implementation rather than the paper: separation boundaries k = 0/1/2/4,
per-class cutoffs Rc = 5/9/13/24 A, intervals 17/30/43/80 at a constant 0.3 A bin,
12 base-typed beads, and
    E = 1.0*E1 + 1.5*E2 + 2.4*E3 + 6.5*E4/fun(N),  fun(N) = -355/sqrt(N+16) + 72.

    energy_binned        - exactly what the C code evaluates (integer-bin lookup)
    energy_forces_interp - the table sampled at bin centres and linearly interpolated,
                           with the analytic gradient; verified self-consistent to 9e-10
                           against central finite differences, at 0.7 percent energy cost

Pair arrays are built once (bead_pair_arrays) so the inner loops stay vectorised; a
per-pair Python loop made the relaxation comparison 1000x too slow.
"""
from pathlib import Path
import numpy as np

import _cgdata
DATA = _cgdata.cgrnasp()
SPEC = [("0-1_short-ranged.potential", 17, 1.0),
        ("1-2_short-ranged.potential", 30, 1.5),
        ("2-4_short-ranged.potential", 43, 2.4),
        ("long-ranged.potential", 80, 6.5)]
BIN = 0.3
BASE_ORDER = "AUCG"          # the order used in data/12atom_type.dat


def load_tables():
    out = []
    for fn, iv, w in SPEC:
        t = np.zeros((12, 12, iv))
        idx = 0
        for line in open(DATA / fn):
            p = line.split()
            if len(p) < 4:
                continue
            blk = idx // iv
            i, j = blk // 12, blk % 12
            k = int(p[2])
            if k < iv:
                t[i, j, k] = float(p[3])
            idx += 1
        out.append((t, iv, w))
    return out


TABLES = load_tables()


def fun(n):
    return -355.0 / np.sqrt(n + 16.0) + 72.0


def bead_types(base_of):
    return np.array([3 * BASE_ORDER.index(b) + k for b in base_of for k in range(3)])


def classify(sep, same_chain=True):
    if not same_chain or sep > 4:
        return 3
    return {1: 0, 2: 1, 3: 2, 4: 2}[sep]


def bead_pair_arrays(L):
    """(bead_a, bead_b, class) as arrays for every inter-residue bead pair."""
    a, b, c = [], [], []
    for i in range(L):
        for j in range(i + 1, L):
            cl = classify(j - i)
            for ka in range(3):
                for kb in range(3):
                    a.append(3 * i + ka)
                    b.append(3 * j + kb)
                    c.append(cl)
    return np.array(a), np.array(b), np.array(c)


def _class_weight(c, N):
    w = SPEC[c][2]
    return w / fun(N) if c == 3 else w


def energy_binned(x, types, PA, PB, PC, N):
    e = 0.0
    r = np.linalg.norm(x[PA] - x[PB], axis=1)
    for ci, (t, iv, w) in enumerate(TABLES):
        m = (PC == ci) & (r < iv * BIN)
        if m.any():
            k = (r[m] / BIN).astype(int)
            e += _class_weight(ci, N) * t[types[PA[m]], types[PB[m]], k].sum()
    return e


def energy_forces_interp(x, types, PA, PB, PC, N, want_forces=True):
    """Linear interpolation between bin centres, with the analytic -dE/dx.

    Distances past the per-class cutoff are skipped, as the C code does, not extrapolated;
    the two ends are held flat.
    """
    f = np.zeros_like(x) if want_forces else None
    e = 0.0
    d = x[PA] - x[PB]
    r = np.linalg.norm(d, axis=1)
    for ci, (t, iv, w) in enumerate(TABLES):
        m = (PC == ci) & (r > 1e-12) & (r < iv * BIN)
        if not m.any():
            continue
        wc = _class_weight(ci, N)
        ta, tb, rm, dm = types[PA[m]], types[PB[m]], r[m], d[m]
        u = rm / BIN - 0.5
        k0 = np.floor(u).astype(int)
        frac = u - k0
        lo = k0 < 0
        hi = k0 >= iv - 1
        kk = np.clip(k0, 0, iv - 2)
        e0 = t[ta, tb, kk]
        e1 = t[ta, tb, np.minimum(kk + 1, iv - 1)]
        frac = np.where(lo, 0.0, np.where(hi, 1.0, frac))
        slope = np.where(lo | hi, 0.0, (e1 - e0) / BIN)
        e += wc * float((e0 + frac * (e1 - e0)).sum())
        if want_forces:
            fmag = (wc * slope / np.maximum(rm, 1e-12))[:, None] * dm
            np.add.at(f, PA[m], -fmag)
            np.add.at(f, PB[m], fmag)
    return e, f
