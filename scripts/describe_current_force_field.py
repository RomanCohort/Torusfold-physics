"""A measured picture of the force field as it currently stands.

Everything in this arc changed constants, forces and one loader, so the energy budget printed
in 3u is stale -- it was taken before K_ANGLE went 600 -> 28.1 and K_DIH 500 -> 7.2, which
scale two of the largest terms by 21x and 69x. This re-measures per-term energy and force on
the current constants, under both pair-weight regimes, so the picture is read rather than
recalled.

Run: python scripts/describe_current_force_field.py [n_structs]
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
W = np.load(CACHE)["w"] if CACHE.exists() else np.full(100, 0.35)

structs = [s for s in B.load_structures(limit=300) if len(s["pairs"]) >= 4][:N]
print(f"{len(structs)} structures; kBT = {B.KBT} kJ/mol at 300 K")
print()


def per_term(s, weights):
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    pw = torch.tensor(weights, dtype=torch.float64)
    F, E = FT.term_energies_forces(pos, ij, pw, cell_list=cl)
    full, _ = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
    return F, E, float(full.reshape(-1)[0])


gen = np.random.default_rng(3)
regimes = {"pair_w = ones": None, "pair_w from the measured RCM shape": W}
for label, sample in regimes.items():
    acc_e = {}
    acc_f = {}
    fulls = []
    for s in structs:
        npr = len(s["pairs"])
        w = np.ones(npr) if sample is None else \
            np.asarray(sample)[gen.integers(0, len(sample), npr)]
        F, E, full = per_term(s, w)
        for k in E:
            acc_e.setdefault(k, []).append(abs(E[k]))
            acc_f.setdefault(k, []).append(np.linalg.norm(F[k], axis=1))
        fulls.append(full)
    print("=" * 88)
    print(label)
    print(f"  mean |full energy| {np.mean(np.abs(fulls)):,.0f} kJ/mol over {len(structs)} "
          f"structures")
    print()
    print(f"  {'term':22s} {'mean |E|':>11s} {'share':>8s} {'p95 |F|':>10s} {'max |F|':>10s} "
          f"{'over cap':>9s}")
    print("  " + "-" * 74)
    tot = np.mean(np.abs(fulls))
    rows = sorted(acc_e, key=lambda k: -np.mean(acc_e[k]))
    for k in rows:
        e = np.mean(acc_e[k])
        f = np.concatenate(acc_f[k])
        print(f"  {k:22s} {e:11.1f} {e/tot*100:7.1f}% {np.percentile(f,95):10.2f} "
              f"{f.max():10.2f} {(f>CAP).mean()*100:8.2f}%")
    print()

# constants as they currently read
print("=" * 88)
print("constants as shipped now")
for name in ("K_BB", "K_INTRA", "K_PAIR", "K_STACK", "K_ANGLE", "K_DIH", "K_CLASH",
             "K_BSJ", "K_BSJ_GUIDE", "K_PAIR_GUIDE", "K_BSJ_CONTACT", "K_BPP", "K_MG"):
    print(f"  {name:16s} {getattr(C, name, float('nan')):>10.3f}")
print()
for name in ("BOND_P_NEXT", "BOND_P_C4", "BOND_C4_N", "PAIR_NN", "STACK_R0",
             "CLASH_DIST", "C_MG_DEFAULT", "C_NA_DEFAULT"):
    print(f"  {name:16s} {getattr(C, name, float('nan')):>10.4f}")
print(f"  {'ANGLE_PPP':16s} {np.degrees(C.ANGLE_PPP):>10.2f} deg  (cos {np.cos(C.ANGLE_PPP):+.3f})")
print(f"  {'DIH_PPPP':16s} {np.degrees(C.DIH_PPPP):>10.2f} deg  (cos {np.cos(C.DIH_PPPP):+.3f})")
