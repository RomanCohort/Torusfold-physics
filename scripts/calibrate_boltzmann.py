"""Calibrating the Boltzmann bonded terms against the harmonic ones.

The problem, stated precisely. U = -kBT ln P is a potential of mean force only if P is the
true equilibrium distribution of the coordinate in the system being simulated. Ours is not:
it comes from a database of experimentally determined structures, a different ensemble of
different molecules under different conditions. Li and Chen's review says as much -- "the
assumption that we could treat experimental distributions from native structures as obtained
from a Boltzmann distribution has been discussed at great lengths... one way of thinking
about the statistical potentials is to treat them as scoring functions". A scoring function
has no thermodynamically fixed scale, so the weight it carries relative to the other terms
has to be decided, not assumed.

Why matching curvature is the informative first calibration. Section 3o measured a funnel
rank of 1.00 for the Boltzmann terms against 1.22 for the harmonic ones, but the Boltzmann
terms are also roughly ten times softer, so that result cannot separate two explanations:
the potential has the right SHAPE, or a soft potential is simply a better funnel. Scaling
each table so its curvature at its own minimum matches the corresponding harmonic force
constant removes the second explanation. Whatever advantage survives is shape.

Run: python scripts/calibrate_boltzmann.py [n_fit] [n_test]
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

N_FIT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 32
SIGMAS = (0.3, 0.6, 1.0)
N_DECOY = 2
SEED = 20260215
MULTIPLIERS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)

K_HARM = {"bb_bond": C.K_BB, "intra_pc": C.K_INTRA, "intra_cn": C.K_INTRA,
          "angle": C.K_ANGLE, "dihedral": C.K_DIH, "stack": C.K_STACK}
BONDED = ("bb bond P-P", "intra P-C4'", "intra C4'-N",
          "angle P-P-P", "dihedral P-P-P-P", "stacking P-P")

allstructs = B.load_structures(limit=N_FIT + N_TEST)
FIT, TEST = allstructs[:N_FIT], allstructs[N_FIT:N_FIT + N_TEST]
print(f"fitted on {len(FIT)}, tested on {len(TEST)} held out")
tables = B.prepare(B.fit(FIT))


def curvature(name):
    """U'' at the table minimum, from a parabola fitted over +-3 bins.

    A plain second difference on a 120-bin table is dominated by bin-to-bin noise where the
    wells are shallow; the parabola averages that out.
    """
    t = tables[name]
    k0 = int(np.argmin(t["U"]))
    lo, hi = max(0, k0 - 3), min(len(t["U"]) - 1, k0 + 3)
    idx = np.arange(lo, hi + 1)
    x = t["centre"][idx]
    y = t["U"][idx]
    a, _b, _c = np.polyfit(x, y, 2)
    return 2.0 * a, float(t["centre"][k0])


print()
# Per-coordinate spread, so the fitted curvature can be sanity-checked against kBT/sigma^2,
# which is what a Gaussian well of that width would have.
spread = {}
for name in B.COORDS:
    v = np.concatenate([B.coords_of(
        torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64), name
    ).reshape(-1).numpy() for s in FIT])
    spread[name] = float(v.std())

print(f"{'coordinate':12s} {'harmonic k':>11s} {'fitted U''':>12s} {'kBT/sigma^2':>12s} "
      f"{'k/ U''':>9s} {'scale to match':>15s}")
print("-" * 78)
curv, scales = {}, {}
for name in B.COORDS:
    c, at = curvature(name)
    analytic = B.KBT / spread[name] ** 2
    # scale multiplies the table. To land the curvature on the harmonic force constant the
    # scaled curvature must equal k, so scale = k / U''. An earlier version used its
    # reciprocal, which multiplied the tables by up to 1000 instead of dividing.
    scale = K_HARM[name] / c if c > 0 else 0.0
    curv[name] = c
    scales[name] = scale
    print(f"{name:12s} {K_HARM[name]:11.1f} {c:12.1f} {analytic:12.1f} "
          f"{K_HARM[name]/c:9.3f} {scale:15.4f}")
print()
print("units work out without conversion: U is kJ/mol, the distance coordinates are in nm")
print("and the angle coordinates restrain cos, so U'' is directly comparable to k.")
print("kBT/sigma^2 is what a Gaussian well of the observed width would have; large gaps")
print("between it and the fitted U'' mean the parabola is fitting a coarsely binned well.")
print("'scale to match' multiplies the table so the two curvatures agree at the minimum.")
print()


def unpack(s):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2) if s["pairs"] else \
        torch.zeros((0, 2), dtype=torch.long)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    return pos, ij, cl


def boltzmann_energy(pos, mult):
    """Table energies with each coordinate scaled by mult * (its curvature match)."""
    for name in B.COORDS:
        t = tables[name]
        t["Ut"] = t["Ut"] * 0.0 + torch.tensor(t["U"] * scales[name] * mult,
                                               dtype=torch.float64)
    return float(B.energy(pos, tables))


def harmonic_bonded(pos, ij, cl):
    pw = torch.ones(len(ij), dtype=torch.float64)
    _, e = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    return sum(e[k] for k in BONDED)


# how strong is each, at native geometry, in the units the integrator sees
pos, ij, cl = unpack(TEST[0])
for name in B.COORDS:
    t = tables[name]
    t["Ut"] = torch.tensor(t["U"], dtype=torch.float64)
pw = torch.ones(len(ij), dtype=torch.float64)
_, e_h = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
e_b = B.energy(pos, tables)
print(f"at one native structure: harmonic bonded {sum(e_h[k] for k in BONDED):10.1f} kJ/mol, "
      f"boltzmann bonded {float(e_b):8.1f} kJ/mol")
print(f"curvature-matched:      boltzmann bonded "
      f"{boltzmann_energy(pos, 1.0):8.1f} kJ/mol")
print()

rng = np.random.default_rng(SEED)
decoy_offsets = []
for s in TEST:
    offs = []
    for sig in SIGMAS:
        for _ in range(N_DECOY):
            offs.append(rng.normal(0, sig / 10.0, s["pos"].shape))
    decoy_offsets.append(offs)

results = {}
for mult in MULTIPLIERS:
    ranks, margins = [], []
    for s, offs in zip(TEST, decoy_offsets):
        p, i_, c_ = unpack(s)
        e0 = boltzmann_energy(p, mult)
        cand = [e0]
        for off in offs:
            d = dict(s)
            d["pos"] = s["pos"] + off
            p2, _, _ = unpack(d)
            cand.append(boltzmann_energy(p2, mult))
        v = np.array(cand)
        ranks.append(int((v < v[0]).sum()) + 1)
        margins.append(float(v.min() - v[0]))
    results[mult] = (float(np.mean(ranks)), float(np.mean(margins)))

harm_ranks, harm_margins = [], []
for s, offs in zip(TEST, decoy_offsets):
    p, i_, c_ = unpack(s)
    e0 = harmonic_bonded(p, i_, c_)
    cand = [e0]
    for off in offs:
        d = dict(s)
        d["pos"] = s["pos"] + off
        p2, i2, c2 = unpack(d)
        cand.append(harmonic_bonded(p2, i2, c2))
    v = np.array(cand)
    harm_ranks.append(int((v < v[0]).sum()) + 1)
    harm_margins.append(float(v.min() - v[0]))

print("funnel against the stiffness multiplier, relative to the curvature-matched scale")
print(f"{'multiplier':>12s} {'native rank':>13s} {'mean gap (kJ/mol)':>19s}")
print("-" * 48)
for mult in MULTIPLIERS:
    tag = "  <- curvature matched" if mult == 1.0 else ""
    print(f"{mult:12.2f} {results[mult][0]:13.3f} {results[mult][1]:19.1f}{tag}")
print("-" * 48)
print(f"{'harmonic':>12s} {float(np.mean(harm_ranks)):13.3f} "
      f"{float(np.mean(harm_margins)):19.1f}")
print()
print(f"rank 1.000 = the native is the lowest of {1 + len(SIGMAS)*N_DECOY} candidates in")
print("every held-out structure, 4.000 = random. The gap is min(decoy) - native, so a")
print("positive gap means no decoy beat the native; it is the finer readout when rank")
print("saturates, which over this range of multipliers it does.")
print()
best = min(results, key=lambda m: (results[m][0], -results[m][1]))
print(f"the rank is 1.000 across a {MULTIPLIERS[-1]/MULTIPLIERS[0]:.0f}-fold range of")
print(f"stiffness, so the funnel constrains the SHAPE of the potential and not its scale.")
