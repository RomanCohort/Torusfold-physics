"""Calibrating bpp: a criterion from data, a grid, and a before/after.

Why bpp is the target. 3z: the whole field put 28.44 percent of beads on the cap after the
angle and dihedral were fixed, and 3x attributes 22.31 of those points to bpp alone. It is
there by construction. E = -K_BPP * w * softplus((1.0 - r)/0.3) has force
K_BPP / 0.3 * w * sigmoid(x), so at the target distance r = 1.0 nm, where x = 0 and
sigmoid(0) = 0.5, every pair experiences K_BPP / 0.6, which is 1000 kJ/mol/nm at the shipped
600 -- five times the cap, whatever the geometry.

The criterion. kBT/sigma^2 does not transfer: softplus is one-sided and has no equilibrium
point, so there is no curvature to match. But two terms restraining the same quantity should
not differ in force scale by a factor of fifty. The harmonic WC pair term's force scale at
the observed geometry is kBT/sigma_NN, so setting

    K_BPP / 0.6  =  kBT / sigma_NN        ->  K_BPP = 0.6 * kBT / sigma_NN

makes the two agree.

The cost side. bpp is the only load-bearing term: 3u found it favouring the native in 23 of
24 structures and 3v in 111 of 111 under register shifts. Softening it must therefore be
checked against the fold signal, not just against the cap.

Run: python scripts/calibrate_bpp.py [n_structs]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import cg_force_terms as FT
import torusfold.scheme2.torch_cgsim as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CAP = 200.0
CACHE = Path(__file__).resolve().parent.parent / "results" / "rcm_weights.npz"
GRID = (3.0, 10.0, 30.0, 100.0, 300.0, 600.0)
MIN_RUN = 3
SEED = 20260222

structs = [s for s in B.load_structures(limit=300) if len(s["pairs"]) >= 6][:N]

# sigma of the N-N distance over the observed pairs
nn = []
for s in structs:
    Nn = s["pos"][:, 2, :]
    for i, j in s["pairs"]:
        nn.append(float(np.linalg.norm(Nn[i] - Nn[j])))
nn = np.array(nn)
sigma = float(nn.std())
crit = 0.6 * B.KBT / sigma
print(f"{len(structs)} structures, {len(nn)} pairs")
print(f"observed N-N distance: mean {nn.mean():.3f} nm, sd {sigma:.3f}")
print(f"criterion: K_BPP = 0.6 * kBT / sigma = 0.6 * {B.KBT} / {sigma:.3f} = {crit:.1f}")
print(f"shipped: {C.K_BPP}, whose force at r = r0 is {C.K_BPP/0.6:.0f} kJ/mol/nm "
      f"({C.K_BPP/0.6/CAP:.1f} times the cap)")
print()


def helix_runs(pairs):
    ps = sorted(pairs, key=lambda p: p[0])
    if not ps:
        return []
    runs, cur = [], [ps[0]]
    for p in ps[1:]:
        a0, b0 = cur[-1]
        a1, b1 = p
        if a1 == a0 + 1 and b1 == b0 - 1:
            cur.append(p)
        else:
            runs.append(cur)
            cur = [p]
    runs.append(cur)
    return [r for r in runs if len(r) >= MIN_RUN]


def shifted(pairs, s):
    out = []
    for run in helix_runs(pairs):
        a = [p[0] for p in run]
        b = [p[1] for p in run]
        for i in range(len(a) - s):
            j = b[i + s]
            if a[i] != j:
                out.append((min(a[i], j), max(a[i], j)))
    return sorted(out) if len(out) >= 3 and len(set(out)) == len(out) else None


usable = [s for s in structs if shifted(s["pairs"], 1)]


def evaluate(s, pairs, w):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    pw = torch.tensor(w, dtype=torch.float64)
    full, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
    _, e = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    return float(full.reshape(-1)[0]), f.reshape(3 * L, 3).numpy(), e


saved = {n: getattr(C, n) for n in ("K_BPP", "_K_BPP") if hasattr(C, n)}
print(f"{'K_BPP':>8s} {'force@r0':>10s} {'on cap (ones)':>14s} {'on cap (RCM)':>13s} "
      f"{'reg-shift wins':>15s}")
print("-" * 66)
try:
    for kb in GRID:
        for n in saved:
            setattr(C, n, kb)
        # whole-field cap saturation under both weight regimes
        caps = []
        for sample in (None, np.load(CACHE)["w"] if CACHE.exists() else np.full(50, 0.35)):
            g = np.random.default_rng(5)
            on, tot = 0, 0
            for s in structs:
                npr = len(s["pairs"])
                w = np.ones(npr) if sample is None else \
                    np.asarray(sample)[g.integers(0, len(sample), npr)]
                _f, f, _e = evaluate(s, s["pairs"], w)
                m = np.linalg.norm(f, axis=1)
                on += int((np.abs(m - CAP) < 1e-6).sum())
                tot += len(m)
            caps.append(on / tot * 100)
        # register-shift fold signal
        wins, n = 0, 0
        g = np.random.default_rng(9)
        for s in usable:
            d = shifted(s["pairs"], 1)
            if d is None:
                continue
            npr = len(s["pairs"])
            w = np.ones(npr)
            e_nat = evaluate(s, s["pairs"], w)[2]
            e_dec = evaluate(s, d, w[:len(d)])[2]
            nat = e_nat["wc pair N-N"] + e_nat["bpp"]
            dec = e_dec["wc pair N-N"] + e_dec["bpp"]
            wins += int(dec > nat)
            n += 1
        print(f"{kb:8.1f} {kb/0.6:10.0f} {caps[0]:13.2f}% {caps[1]:12.2f}% "
              f"{wins:8d}/{n:<5d}")
finally:
    for n, v in saved.items():
        setattr(C, n, v)
print("-" * 66)
print()
print("force@r0 is the force this term applies to every pair at its own target distance.")
print("reg-shift wins counts the near-native decoy comparisons where the correct register")
print("scores lower; it is the harness where bpp was decisive at 111/111.")
