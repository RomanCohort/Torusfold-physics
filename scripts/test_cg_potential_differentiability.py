"""Make the cgRNASP tables differentiable, then check F = -dE/dr against finite differences.

Reference   : the binned lookup the C implementation uses, potential[i][j][int(r/0.3)].
Interpolated: the same table sampled at bin centres and linearly interpolated, which has a
              piecewise-constant derivative and therefore a well-defined force.

The test is deliberately narrow: it asks whether the interpolated representation is
self-consistent (analytic gradient == finite difference), and how badly the binned one
fails the same test. It says nothing about whether the potential is any good.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import truth_1ehz

DATA = Path(r"D:\torusfold-cgdata\cgRNASP\cgRNASP\data")
SPEC = [("0-1_short-ranged.potential", 17, 1.0),      # separation == 1
        ("1-2_short-ranged.potential", 30, 1.5),      # separation == 2
        ("2-4_short-ranged.potential", 43, 2.4),      # separation 3-4
        ("long-ranged.potential", 80, 6.5)]           # separation >= 5, or another chain
BIN = 0.3
BASE_ORDER = "AUCG"                                    # 12atom_type.dat order


def load_tables():
    out = []
    for fn, iv, w in SPEC:
        # read_potential() in the C source discards the first three integers and fills
        # p[n1][n2][n3] by loop position, so the block order is what matters here.
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
        expected = 144 * iv
        if idx != expected:
            raise SystemExit(f"{fn}: read {idx} lines, expected {expected}")
        out.append((t, iv, w))
    return out


TABLES = load_tables()


def fun(n):
    return -355.0 / np.sqrt(n + 16.0) + 72.0


def bead_types(base_of):
    return [3 * BASE_ORDER.index(b) + k for b in base_of for k in range(3)]


def classify(sep, same_chain):
    if not same_chain or sep > 4:
        return 3
    return {1: 0, 2: 1, 3: 2, 4: 2}[sep]


def pairs_of(beads_flat, types, L):
    """(bead_a, bead_b, type_a, type_b, class) for every inter-residue bead pair."""
    out = []
    for i in range(L):
        for j in range(i + 1, L):
            c = classify(j - i, True)
            for ka in range(3):
                for kb in range(3):
                    out.append((3 * i + ka, 3 * j + kb,
                                types[3 * i + ka], types[3 * j + kb], c))
    return out


def _class_weight(c, N):
    """Per-class weight, including the 1/fun(N) normalisation on the long-range term."""
    w = SPEC[c][2]
    return w / fun(N) if c == 3 else w


def energy_binned(x, P, N):
    """Exactly what the C implementation evaluates: an integer-bin lookup."""
    e = 0.0
    for a, b, ta, tb, c in P:
        t, iv, _ = TABLES[c]
        r = float(np.linalg.norm(x[a] - x[b]))
        k = int(r / BIN)
        if k < iv:
            e += _class_weight(c, N) * t[ta, tb, k]
    return e


def energy_forces_interp(x, P, N):
    """Linear interpolation between bin centres, with the analytic -dE/dx.

    Distances past the per-class cutoff are skipped, exactly as the C code does - not
    extrapolated. At the two ends the potential is held flat, so the slope there is zero.
    """
    f = np.zeros_like(x)
    e = 0.0
    for a, b, ta, tb, c in P:
        t, iv, _ = TABLES[c]
        d = x[a] - x[b]
        r = float(np.linalg.norm(d))
        if r < 1e-12 or r >= iv * BIN:
            continue
        w = _class_weight(c, N)
        u = r / BIN - 0.5
        k0 = int(np.floor(u))
        if k0 < 0:
            k0, frac, slope = 0, 0.0, 0.0
        elif k0 >= iv - 1:
            k0, frac, slope = iv - 2, 1.0, 0.0
        else:
            frac = u - k0
            slope = (t[ta, tb, k0 + 1] - t[ta, tb, k0]) / BIN
        e += w * (t[ta, tb, k0] + frac * (t[ta, tb, k0 + 1] - t[ta, tb, k0]))
        f[a] += -w * slope * (d / r)
        f[b] += w * slope * (d / r)
    return e, f


order, res, base_of, ps, partner, meta = truth_1ehz.load()
n = 20
base = base_of[:n]
L = n
beads = np.zeros((3 * L, 3))
for i in range(L):
    r = res[order[i]]
    gly = "N9" if base[i] in "AG" else "N1"
    beads[3 * i] = r["P"]
    beads[3 * i + 1] = r["C4'"]
    beads[3 * i + 2] = r[gly]
types = bead_types(base)
P = pairs_of(beads, types, L)
print(f"test system: {L} residues, {len(P)} bead pairs   fun(N) = {fun(L):.2f}")
print()

e_bin = energy_binned(beads, P, L)
e_int, f_int = energy_forces_interp(beads, P, L)
print(f"binned energy       : {e_bin:12.4f} kBT")
print(f"interpolated energy : {e_int:12.4f} kBT")
print(f"difference          : {e_int - e_bin:+12.4f} kBT")
print()

h = 1e-4
print(f"analytic gradient vs central finite differences (h = {h} A)")
ok = tot = 0
worst = 0.0
rng = np.random.default_rng(0)
for a in rng.choice(len(beads), size=12, replace=False):
    for ax in (0, 1, 2):
        xp = beads.copy(); xp[a, ax] += h
        xm = beads.copy(); xm[a, ax] -= h
        fd = (energy_forces_interp(xp, P, L)[0] - energy_forces_interp(xm, P, L)[0]) / (2 * h)
        an = -f_int[a, ax]
        tot += 1
        if abs(fd - an) <= 1e-6 + 1e-4 * abs(an):
            ok += 1
        worst = max(worst, abs(fd - an))
print(f"  interpolated : {ok}/{tot} components agree, worst |FD - analytic| = {worst:.3e}")
print()

zero = tot2 = 0
vals = []
for a in rng.choice(len(beads), size=12, replace=False):
    for ax in (0, 1, 2):
        xp = beads.copy(); xp[a, ax] += h
        xm = beads.copy(); xm[a, ax] -= h
        fd = (energy_binned(xp, P, L) - energy_binned(xm, P, L)) / (2 * h)
        vals.append(abs(fd)); tot2 += 1
        if fd == 0.0:
            zero += 1
print(f"  binned       : {zero}/{tot2} components have FD exactly 0, "
      f"max |FD| = {np.max(vals):.3e}")
print()
print("So the interpolated representation is self-consistent, and the binned one has no")
print("usable gradient at all at this step size.")
