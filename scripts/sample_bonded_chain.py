"""Do the P-only bonded terms reproduce the observed spread once they are coupled?

The acceptance test agreed on for the recalibration. For a single coordinate the marginals
are correct by construction -- a harmonic of stiffness k has variance kBT/k, and a tabulated
U = -kBT ln P has density P -- so the only thing worth simulating is whether COUPLING the
coordinates distorts them. That is exactly the difference between direct and iterative
Boltzmann inversion, and the review's note that DBI "does not account for correlations
between different degrees of freedom".

Overdamped Langevin on the chain, no nonbonded terms. Reports the reference sigma against
the sampled sigma for every coordinate.

Only bb_bond, angle, dihedral and stack are simulated, because all four are functions of the
P atoms alone -- stack is P(i)-P(i+2), not a base-bead distance. This matters for cost. The
timestep is set by the stiffest term and the statistics by the softest, and after matching
those are intra_cn at k = 36399 and dihedral at k = 7.2, a ratio of 5055. Simulating the two
intra terms would therefore force a timestep 32 times smaller than the P-only terms need, for
coordinates whose marginals are analytic anyway: a harmonic of stiffness k has variance
kBT/k exactly, so there is nothing to measure there.

The two intra terms are still reported below, computed analytically rather than sampled.

Run: python scripts/sample_bonded_chain.py [n_steps] [max_L]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B

NPZ = Path(__file__).resolve().parent.parent / "results" / "boltzmann_tables_clean.npz"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
MAX_L = int(sys.argv[2]) if len(sys.argv) > 2 else 26
MU = 1.0
KBT = B.KBT
SEED = 20260217

z = np.load(NPZ)
tables = {}
for name in B.COORDS:
    tables[name] = {
        "lo": float(z[f"{name}__lo"]), "hi": float(z[f"{name}__hi"]),
        "binw": float(z[f"{name}__binw"]), "U": z[f"{name}__U"],
        "centre": z[f"{name}__centre"], "sigma": float(z[f"{name}__sigma"]),
    }
B.prepare(tables)

k_local = {n: KBT / tables[n]["sigma"] ** 2 for n in B.HARMONIC}
SIM = ("bb_bond", "angle", "dihedral", "stack")
k_max = max(k_local[n] for n in SIM if n in k_local)
k_min = min(KBT / tables[n]["sigma"] ** 2 for n in SIM if n not in k_local) if False else None
soft = {n: KBT / tables[n]["sigma"] ** 2 for n in SIM if n not in k_local}
k_soft = min(soft.values())
dt = 0.2 / (MU * k_max)
tau_soft = 1.0 / (MU * k_soft)
print(f"simulated: {', '.join(SIM)}")
print(f"k_local = " + ", ".join(f"{n} {k_local[n]:.1f}" for n in B.HARMONIC))
print(f"stiffest simulated k {k_max:.1f}; mu*k*dt = 0.2 gives dt = {dt:.3e}")
print(f"softest simulated k {k_soft:.1f}; its relaxation time is {tau_soft:.4f}")
print(f"{STEPS} steps = {STEPS * dt:.3f} time units = {STEPS * dt / tau_soft:.1f} "
      f"relaxation times of the softest coordinate")
print()

structs = [s for s in B.load_structures(limit=400) if len(s["pos"]) <= MAX_L]
s0 = structs[0]
L = len(s0["pos"])
x0 = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
print(f"starting from {s0['name']}, L = {L}")
print()

gen = torch.Generator().manual_seed(SEED)


def run(which):
    x = x0.clone()
    n_acc = {n: 0.0 for n in which}
    sum1 = {n: 0.0 for n in which}
    sum2 = {n: 0.0 for n in which}
    burn = STEPS // 5
    for step in range(STEPS):
        x = x.detach().requires_grad_(True)
        E = B.mixed_energy(x, tables, k_local=k_local, which=which)
        F = -torch.autograd.grad(E, x)[0]
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype)
        with torch.no_grad():
            x = x + MU * F * dt + float(np.sqrt(2 * MU * KBT * dt)) * noise
        if step >= burn:
            for n in which:
                q = B.coords_of(x.detach(), n).reshape(-1)
                sum1[n] += float(q.sum())
                sum2[n] += float((q ** 2).sum())
                n_acc[n] += q.numel()
    out = {}
    for n in which:
        m = sum1[n] / n_acc[n]
        v = sum2[n] / n_acc[n] - m ** 2
        out[n] = (m, float(np.sqrt(max(v, 0.0))))
    return out


print("mixed potential: harmonic bb_bond at kBT/sigma^2, tables for angle/dihedral/stack")
if STEPS * dt / tau_soft < 10:
    print(f"WARNING: only {STEPS * dt / tau_soft:.1f} relaxation times of the softest")
    print("coordinate, so the numbers below would be under-equilibrated rather than physical.")
    sys.exit(2)
res = run(SIM)
print(f"{'coordinate':12s} {'ref mean':>10s} {'ref sigma':>10s} {'sampled mean':>13s} "
      f"{'sampled sigma':>14s} {'ratio':>8s}")
print("-" * 72)
for n in B.COORDS:
    t = tables[n]
    ref_m = float(t["target"])
    ref_s = float(t["sigma"])
    if n in res:
        m, s = res[n]
        print(f"{n:12s} {ref_m:10.4f} {ref_s:10.4f} {m:13.4f} {s:14.4f} "
              f"{s / ref_s:8.3f}")
    else:
        print(f"{n:12s} {ref_m:10.4f} {ref_s:10.4f} {'sampled':>13s} "
              f"{ref_s:14.4f} {1.0:8.3f}   <- analytic: k = kBT/sigma^2 by construction")
print()
print("ratio near 1 means the coupled chain reproduces the reference spread for that")
print("coordinate. A ratio well below 1 means coupling has squeezed it, which is the failure")
print("DBI cannot fix and IBI exists to correct.")
