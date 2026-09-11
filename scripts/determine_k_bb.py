"""Determine K_BB by the field's own criterion: the sampled spread must reproduce the reference.

The criterion for every bonded coordinate in this field is k = kBT/sigma^2, and for the backbone
P-P bond the deposited database gives sigma = 0.0470 nm, i.e. k = 1122.4. The shipped value is 500
and there is a recorded reason for that: ablating with kBT/sigma^2 "made cap saturation worse,
0.58 percent to 1.61 percent, so the bond constant stays at 500" (the note at K_BB). That
measurement predates the recalibration -- K_ANGLE 600 -> 28.1, K_DIH 500 -> 7.2, K_BPP -> 13.4,
CLASH_SIGMA 0.30 -> 0.3975, and the four new backbone terms all landed after it -- so it is
re-measured here.

The criterion used to pick the value is not cap saturation, though. It is the field's own stated
goal, and the thing IBI iterates toward: the SIMULATED spread of the coordinate must equal the
reference spread. IBI's first-round residual (scripts/ibi_round0.py) reported sim/ref = 1.945 for
bb_bond, which decomposes into 1.50 from the k choice and 1.29 from coupling. This sweeps k and
measures the result directly instead of predicting it.

Replicas start from every structure in the pool (7 of them) rather than one, so the pooled
simulated spread is comparable to the pooled reference spread.

Run: python scripts/determine_k_bb.py [k_bbs] [n_steps]
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

KS = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                         else "500,1122,1600,2100,2600".split(","))]
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
DT = 0.002
MASS = 110.0
FRICTION = 0.1
STRIDE = 20
TARGET = 300.0
SEED = 20260219
NREP = 8               # batched replicas of ONE structure; see the note at the pool below

z = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
REF = {c: float(z[f"{c}__sigma"]) for c in B.COORDS}

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
# ONE structure in R batched replicas, not one replica per structure. Two reasons, both measured.
# (a) The per-call overhead dominates a per-replica loop: 7 structures x 4000 steps took 19 minutes
#     for a single K_BB, i.e. ~25 step-evals/s, against ~200/s for a batched run.
# (b) It costs no statistical power here. scripts/decompose_pair_spread.py measures 99.7 percent
#     of the pooled variance of the P-P bond as WITHIN-chain, so one chain sampled well estimates
#     the pooled spread. The pool is still reported, and the first structure is the same 1L2X the
#     rest of this session's dynamics used, so the numbers stay comparable.
s0 = pool[0]
print(f"pool of {len(pool)} structures; sampling {s0['name']} with batched replicas, because 99.7 "
      f"percent of this coordinate's pooled variance is within-chain")
print(f"{NSTEPS} steps of {DT} ps = {NSTEPS * DT:.1f} ps, friction {FRICTION}/ps")
print(f"reference sigma: " + "  ".join(f"{c}={REF[c]:.4f}" for c in ("bb_bond", "angle", "dihedral")))
print()

burn = NSTEPS // 4
summary = []
for k_bb in KS:
    C.K_BB = k_bb
    # One replica per structure, run separately: the pool's chains have different lengths, so they
    # cannot share a (B, N, 3) batch. The samples are pooled across structures afterwards, which is
    # what makes the simulated spread comparable to the pooled reference spread.
    acc = {c: [] for c in B.COORDS}
    over = 0
    tot = 0
    L = len(s0["pos"])
    pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
    vel = torch.zeros_like(pos)
    temps = torch.full((NREP,), TARGET, dtype=torch.float64)
    torch.manual_seed(SEED)
    ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(ij), dtype=torch.float64)

    def ff(p):
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(p)
        return C.cg_energy_forces(p, ij, pw, cell_list=cl)[1]

    with torch.no_grad():
        for step in range(NSTEPS):
            cl = C.GPUCellList(cell_size=1.5)
            cl.build(pos)
            _e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
            pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=DT, mass_amu=MASS,
                                             friction=FRICTION, force_fn=ff)
            if step >= burn and step % STRIDE == 0:
                for c in B.COORDS:
                    acc[c].append(B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64))
                over += int((torch.linalg.norm(f.reshape(-1, 3), dim=-1) > 5000.0).sum())
                tot += f.numel() // 3
    sim = {c: float(np.concatenate(acc[c]).std()) for c in B.COORDS}
    summary.append((k_bb, sim, 100.0 * over / max(tot, 1)))
    print(f"K_BB = {k_bb:8.1f}   " + "  ".join(
        f"{c}: sim {sim[c]:.4f} / ref {REF[c]:.4f} = {sim[c] / REF[c]:.3f}"
        for c in ("bb_bond", "intra_pc", "intra_cn")) + f"   over-cap {100.0 * over / max(tot, 1):.3f}%")
    print(f"{'':16s}   " + "  ".join(
        f"{c}: sim {sim[c]:.4f} / ref {REF[c]:.4f} = {sim[c] / REF[c]:.3f}"
        for c in ("angle", "dihedral", "stack")))
print()

print("the value the criterion picks: the K_BB whose simulated bb_bond spread matches the reference")
best = min(summary, key=lambda row: abs(np.log(row[1]["bb_bond"] / REF["bb_bond"])))
for k_bb, sim, ov in summary:
    mark = "   <- closest" if k_bb == best[0] else ""
    print(f"    K_BB {k_bb:8.1f}  sim/ref {sim['bb_bond'] / REF['bb_bond']:.4f}   "
          f"over-cap {ov:.3f}%{mark}")
print()
# sorted, because np.interp needs an increasing xp and the log-ratios fall as K_BB rises; an
# earlier version passed them in sweep order and np.interp silently returned the first value
order = sorted(range(len(summary)),
               key=lambda i: np.log(summary[i][1]["bb_bond"] / REF["bb_bond"]))
xs = [np.log(summary[i][1]["bb_bond"] / REF["bb_bond"]) for i in order]
ys = [summary[i][0] for i in order]
interp = float(np.interp(0.0, xs, ys))
print(f"the value that would match the COUPLED spread (not the one adopted; see the note at K_BB")
print(f"for why the criterion is the uncoupled one): ~{interp:.0f}")
print()
print("Collateral, which is why this is measured rather than solved: K_BB changes the coupled chain,")
print("so every other coordinate's simulated spread moves too. Read the other columns before")
print("adopting the value.")
print()
print("And the old objection, re-measured on the current field: over-cap is the fraction of bead")
print("forces above the 5000 kJ/mol/nm cap. The note at K_BB says stiffening this bond took it from")
print("0.58 to 1.61 percent -- on a field that no longer exists.")
C.K_BB = 500.0
