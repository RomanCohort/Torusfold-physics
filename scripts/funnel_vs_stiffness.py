"""Can the bonded stiffness be lowered without losing the energy funnel?

3q proposed changing the dihedral restraint from a function of cos(phi) to a function of
phi, on the theory that the cos form goes singular near phi = 0. That theory is wrong twice
over: dE/dx = k(cos(phi)-c)(-sin(phi))dphi/dx puts sin(phi) in the numerator where it
cancels, and the measured correlation between |F| and 1/sin(dihedral) is 0.08. The
divergence of dphi/dx is at collinear bonds, which BOTH forms share, so changing the
functional form cannot remove it.

What the measurement does say (verify_gap_fix_and_dihedral.py, 2941 windows on cleaned
data) is that the stiffness itself is the problem: the median dihedral force is 2323.7
kJ/mol/nm, eleven times the 200 cap, and 94.8 percent of windows exceed it. A force that is
essentially always capped is not a force law.

So the question is whether the stiffness can come down. Section 3p showed the funnel is
blind to the scale of the Boltzmann tables over three orders of magnitude; this asks the
same of the harmonic constants, which is what would actually be changed.

Run: python scripts/funnel_vs_stiffness.py [n_fit] [n_test]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

N_FIT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 32
SIGMAS = (0.3, 0.6, 1.0)
N_DECOY = 2
SEED = 20260216
FACTORS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

KNAMES = ["K_BB", "K_INTRA", "K_ANGLE", "K_DIH", "K_STACK", "K_PAIR",
          "K_PAIR_GUIDE", "K_BSJ", "K_BSJ_GUIDE", "K_BSJ_CONTACT", "K_BPP"]
BONDED = ("bb bond P-P", "intra P-C4'", "intra C4'-N",
          "angle P-P-P", "dihedral P-P-P-P", "stacking P-P")

allstructs = B.load_structures(limit=N_FIT + N_TEST)
FIT, TEST = allstructs[:N_FIT], allstructs[N_FIT:N_FIT + N_TEST]
print(f"{len(FIT)} fit, {len(TEST)} held out")
base_k = {n: getattr(C, n) for n in KNAMES}


def unpack(s):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2) if s["pairs"] else \
        torch.zeros((0, 2), dtype=torch.long)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    return pos, ij, cl


import cg_force_terms as FT


def bonded_energy(s):
    pos, ij, cl = unpack(s)
    pw = torch.ones(len(ij), dtype=torch.float64)
    _, e = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    return sum(e[k] for k in BONDED)


def force_scale(s):
    """Median per-bead |F| over the bonded terms, before the path's cap is applied."""
    pos, ij, cl = unpack(s)
    pw = torch.ones(len(ij), dtype=torch.float64)
    F, _ = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    tot = sum(F[k] for k in BONDED)
    return float(np.median(np.linalg.norm(tot, axis=1)))


rng = np.random.default_rng(SEED)
offsets = []
for s in TEST:
    o = []
    for sig in SIGMAS:
        for _ in range(N_DECOY):
            o.append(rng.normal(0, sig / 10.0, s["pos"].shape))
    offsets.append(o)

print()
print(f"{'factor':>8s} {'FIT funnel rank':>16s} {'TEST funnel rank':>17s} "
      f"{'median |F| (fit)':>17s}")
print("-" * 64)
for fac in FACTORS:
    try:
        for n in KNAMES:
            setattr(C, n, base_k[n] * fac)
        ranks = {}
        for split, data in (("fit", FIT), ("test", TEST)):
            offs = []
            r2 = np.random.default_rng(SEED)
            for s in data:
                o = []
                for sig in SIGMAS:
                    for _ in range(N_DECOY):
                        o.append(r2.normal(0, sig / 10.0, s["pos"].shape))
                offs.append(o)
            rr = []
            for s, o in zip(data, offs):
                e0 = bonded_energy(s)
                cand = [e0]
                for off in o:
                    d = dict(s)
                    d["pos"] = s["pos"] + off
                    cand.append(bonded_energy(d))
                v = np.array(cand)
                rr.append(int((v < v[0]).sum()) + 1)
            ranks[split] = float(np.mean(rr))
        med = float(np.median([force_scale(s) for s in FIT[:8]]))
        print(f"{fac:8.3f} {ranks['fit']:16.3f} {ranks['test']:17.3f} {med:17.1f}")
    finally:
        for n in KNAMES:
            setattr(C, n, base_k[n])
print("-" * 64)
print()
print("rank 1.000 = the native is the lowest of 7 candidates in every structure.")
print("median |F| is what the integrator sees before the 200 kJ/mol/nm cap in")
print("cg_energy_forces, which fires whenever it exceeds 200.")
