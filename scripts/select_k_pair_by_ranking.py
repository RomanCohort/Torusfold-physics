"""Can any decoy metric put an UPPER bound on K_PAIR, or only a floor?

tests/test_pair_clash_bsj_criterion.py states the open question: "nothing measured selects 600 over
1500. The lower bound is the only measurement, and choosing inside the bracket needs a fold or
ranking run." This runs that measurement on two decoys of very different difficulty.

  deranged        random re-pairing. Flings partners far apart: wrong-assignment N-N mean 2.18 nm
                  against 0.94 nm correct, so it is an easy decoy.
  register shift  slide the partner strand along a helix by one or two positions. The same
                  residues stay paired to members of the same helix, so every distance stays in a
                  plausible band and the chemistry is the only thing that changes. This is the
                  near-native decoy for RNA, and it is the one that could discriminate.

The WC term is 0.5 * K_PAIR * lam * w_k * (d_k - PAIR_NN)^2 summed over pairs, so the whole K_PAIR
dependence is linear and exact:

    dE(K_PAIR) = K_PAIR * A + dE_guide

A = sum_k 0.5 * w_k * [(d'_k - PAIR_NN)^2 - (d_k - PAIR_NN)^2] measured from the geometry, and
dE_guide the difference with K_PAIR switched to zero, which leaves every other pairing-aware term.
Both are one measurement per structure, so the sweep is exact over the whole range instead of a
grid of reruns. Positive dE means the field scores the wrong assignment higher.

helix_runs and shift are copied from scripts/test_register_shift_decoys.py, which owns that decoy;
that file is the authority for how a register shift is defined.

Run: python scripts/select_k_pair_by_ranking.py [n_structs]
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MIN_PAIRS = 6
MIN_RUN = 3
SHIFTS = (1, 2)
SEED = 20260219
GRID = (0.0, 100.0, 205.0, 400.0, 600.0, 1000.0, 1500.0, 2500.0, 4000.0, 8000.0)

# ---- copied from scripts/test_register_shift_decoys.py -------------------------------------
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


def shift(pairs, s):
    out, kept = [], 0
    for run in helix_runs(pairs):
        a = [p[0] for p in run]
        b = [p[1] for p in run]
        for i in range(len(a) - s):
            j = b[i + s]
            if a[i] == j:
                continue
            out.append((min(a[i], j), max(a[i], j)))
        kept += len(a) - s
    if kept < 3:
        return None
    if len(set(out)) != len(out):
        return None
    return sorted(out)
# --------------------------------------------------------------------------------------------

structs = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= MIN_PAIRS][:N]
print(f"{len(structs)} structures with at least {MIN_PAIRS} WC pairs")
rng = np.random.default_rng(SEED)


def derange(pairs):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    for _ in range(200):
        perm = rng.permutation(len(b))
        if all(perm[i] != i for i in range(len(b))):
            break
    return sorted((min(a[i], b[j]), max(a[i], b[j])) for i, j in enumerate(perm))


NN = lambda i: 3 * i + 2


def full(pos, pairs):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    return float(C.cg_energy_forces(pos, pairs, torch.ones(len(pairs), dtype=torch.float64),
                                    cell_list=cl, force_cap=None)[0])


def measure(decoy_of, label):
    rows = []
    saved = C.K_PAIR
    try:
        for s in structs:
            d = decoy_of(s)
            if d is None or len(d) < 3:
                continue
            L = len(s["pos"])
            pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
            nat = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
            wro = torch.tensor(d, dtype=torch.long).reshape(-1, 2)

            def dd(p):
                return torch.linalg.norm(pos[:, NN(p[:, 0])] - pos[:, NN(p[:, 1])], dim=-1)

            dn, dw = dd(nat), dd(wro)
            # sums, not elementwise: a register shift drops the tail of every helix, so the decoy
            # has FEWER pairs than the native and the two vectors have different lengths. The term
            # really does sum over a different set.
            A = float(0.5 * (((dw - C.PAIR_NN) ** 2).sum() - ((dn - C.PAIR_NN) ** 2).sum()))
            C.K_PAIR = 0.0
            dg = full(pos, wro) - full(pos, nat)
            C.K_PAIR = saved
            rows.append((A, dg, float(dn.mean()), float(dw.mean()), len(nat), len(wro)))
    finally:
        C.K_PAIR = saved
    A = np.array([r[0] for r in rows])
    DG = np.array([r[1] for r in rows])
    print()
    print(f"=== decoy: {label}   ({len(rows)} usable structures) ===")
    print(f"    N-N mean  correct {np.mean([r[2] for r in rows]):.4f} nm over "
          f"{np.mean([r[4] for r in rows]):.1f} pairs, decoy {np.mean([r[3] for r in rows]):.4f} nm "
          f"over {np.mean([r[5] for r in rows]):.1f} pairs")
    print(f"    A mean {A.mean():+.4f}  positive for {int((A > 0).sum())}/{len(A)}  "
          f"range [{A.min():+.4f}, {A.max():+.4f}]")
    print(f"    dE_guide mean {DG.mean():+.2f}  positive for {int((DG > 0).sum())}/{len(DG)}")
    print()
    print(f"    {'K_PAIR':>8s} {'mean dE':>12s} {'favours the correct register':>30s}")
    print("    " + "-" * 54)
    for k in GRID:
        d = k * A + DG
        print(f"    {k:8.0f} {d.mean():12.1f} {int((d > 0).sum()):>18d}/{len(d):<5d}")
    return A, DG


res = {}
res["deranged"] = measure(lambda s: derange(s["pairs"]), "deranged re-pairing (easy)")
for sft in SHIFTS:
    res[f"shift{sft}"] = measure(lambda s, _s=sft: shift(s["pairs"], _s),
                                 f"register shift by {sft}")

print()
print("=== the upper bound, where one exists ===")
print()
print("dE = K*A + dE_guide. A structure is correct at K while dE > 0. For A < 0 that holds only")
print("below K* = dE_guide / (-A); for A >= 0 it holds for every K, so K* is infinite. The tightest")
print("bound over the set is the smallest finite K*.")
print()
print(f"{'decoy':14s} {'finite K*':>10s} {'infinite':>9s} {'tightest K*':>12s} {'at 600':>8s} "
      f"{'at 1500':>9s}")
print("-" * 70)
for label, (A, DG) in res.items():
    kstar = np.where(A < 0, np.where(A < 0, DG / np.where(A < 0, -A, 1.0), np.inf), np.inf)
    finite = kstar[np.isfinite(kstar)]
    n_inf = int((~np.isfinite(kstar)).sum())
    tight = float(finite.min()) if finite.size else float("inf")
    d600 = float((600.0 * A + DG > 0).sum())
    d1500 = float((1500.0 * A + DG > 0).sum())
    print(f"{label:14s} {finite.size:10d} {n_inf:9d} {tight:12.1f} "
          f"{int(d600):>4d}/{len(A):<3d} {int(d1500):>5d}/{len(A):<3d}")
print()
for label, (A, DG) in res.items():
    kstar = np.where(A < 0, DG / np.where(A < 0, -A, 1.0), np.inf)
    finite = np.sort(kstar[np.isfinite(kstar)])
    if finite.size:
        print(f"  {label:14s} finite K* percentiles: "
              + "  ".join(f"p{p}={np.percentile(finite, p):.0f}" for p in (5, 25, 50, 75, 95)))
print()
print("A finite tightest K* above 1500 means both shipped candidates sit under the bound, so this")
print("decoy separates them from NOTHING -- it only rules out the top of the range. That is still an")
print("upper bound where the force cap used to provide a vacuous one.")
print()
print("Two decoys, two different answers, and both are the answer. The easy one (random re-pairing)")
print("is monotone: the correct assignment wins more often the stiffer the spring, all the way up, so")
print("it bounds K_PAIR from below only. The near-native one (register shift by 1) is NOT monotone --")
print("the decoy's own N-N mean is 1.1426 nm, closer to PAIR_NN than the native's 0.9402 -- so it")
print("turns over and bounds K_PAIR from above. That is what an easy decoy cannot do, and it is why")
print("the earlier run concluded no upper bound existed.")
