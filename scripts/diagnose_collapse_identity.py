"""Which two beads interpenetrate when the chain comes apart?

The clash watch in scripts/ibi_round0.py reports only the closest distance. That is not enough
to decide what to fix: the terms that can pull two beads together act on different bead types.

  N(i)-N(j)   the WC pair restraint, E = 0.5*k_e*(r-1.0)^2 with k_e = K_PAIR*lambda*w. At r = 0.3
              the force is 600*0.70 = 420 kJ/mol/nm and at r = 0 it is 600.
  P(i)-P(j)   no pairing term at all; the pull would have to come from GB/SA or the guides.
  the clash spring is a one-sided linear spring, E = 0.5*500*(0.300-d)^2, so its force at full
              overlap is only 500*0.300 = 150 kJ/mol/nm and it never grows beyond that.

So if the collapsing pair is N-N the pair restraint out-pulls the excluded volume by 4x and the
fix is the repulsion; if it is P-P that story is wrong and the pull is somewhere else.

This runs the shipped field with the corrected constants and records the identity of the closest
pair at every sampled step.

Run: python scripts/diagnose_collapse_identity.py [n_rep] [n_steps]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
# third argument: the force cap, or "none". The cap is applied to the summed force per bead, so
# it clips a diverging repulsion along with everything else; whether the new excluded volume can
# do its job at all depends on this number.
CAPARG = sys.argv[3] if len(sys.argv) > 3 else "200"
CAP = None if CAPARG.lower() == "none" else float(CAPARG)
STRIDE = 25
NAME = ("P", "C4'", "N")
KBT = 2.494

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
temps = torch.full((NREP,), 300.0, dtype=torch.float64)

print(f"structure {s0['name']} L={L} pairs={len(ij)}, {NREP} replicas x {NSTEPS} steps")
print("field: " + "  ".join(f"{n}={getattr(C, n)}" for n in
                            ("K_BB", "K_INTRA_PC", "K_INTRA_CN", "K_PAIR", "K_CLASH")))
print(f"force_cap = {CAP}")
print(f"clash spring: k={C.K_CLASH}, d0={C.CLASH_DIST}. Force at full overlap = "
      f"{C.K_CLASH * C.CLASH_DIST:.0f} kJ/mol/nm")
print(f"pair restraint: k_e={C.K_PAIR}*w. Force at r=0.3 is {C.K_PAIR * 0.7:.0f}, at r=0 is "
      f"{C.K_PAIR:.0f}")
print()

pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
torch.manual_seed(20260219)


def _forces_at(p):
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl2, force_cap=CAP)[1]


pair_type_count = Counter()
min_per_class = {}
steps_below = 0
n_sampled = 0
burn = NSTEPS // 5
for step in range(NSTEPS):
    with torch.no_grad():
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl, force_cap=CAP)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=0.002,
                                         mass_amu=110.0, friction=0.1, force_fn=_forces_at)
    if step >= burn and step % STRIDE == 0:
        with torch.no_grad():
            beads = pos.reshape(NREP, -1, 3)
            n = beads.shape[1]
            dd = torch.cdist(beads, beads) + torch.eye(n, device=beads.device) * 10.0
            flat = dd.reshape(NREP, -1).argmin(dim=1)
            nb = n * n
            for b in range(NREP):
                k = int(flat[b])
                a, c = divmod(k, n)
                t1, t2 = NAME[a % 3], NAME[c % 3]
                label = "-".join(sorted((t1, t2)))
                pair_type_count[label] += 1
                d = float(dd[b, a, c])
                if label not in min_per_class or d < min_per_class[label]:
                    min_per_class[label] = d
            n_sampled += 1
            if float(dd.min()) < C.CLASH_DIST:
                steps_below += 1

# The counts are accumulated per replica, so the denominator is samples TIMES replicas. Dividing
# by the sample count alone makes the shares sum to NREP hundred percent -- which is exactly the
# bug this line replaced.
n_frames = n_sampled * NREP
print(f"sampled {n_sampled} steps x {NREP} replicas = {n_frames} replica-frames")
print()
print("=== which bead-type pair is the closest one, per frame ===")
print(f"{'type pair':10s} {'frames':>8s} {'share':>8s} {'smallest seen (nm)':>20s}")
print("-" * 50)
for label, cnt in pair_type_count.most_common():
    print(f"{label:10s} {cnt:8d} {100 * cnt / n_frames:7.1f}% "
          f"{min_per_class[label]:20.4f}")
print()
print(f"steps with the closest pair below the {C.CLASH_DIST} cutoff: {steps_below}/{n_sampled} "
      f"(the minimum over all replicas, so this is not the per-frame count)")
print()
print("Reading it: N-N means the WC pair restraint is the pull, and its force at r=0.3 is "
      f"{C.K_PAIR * 0.7:.0f} against a clash spring that cannot exceed {C.K_CLASH * C.CLASH_DIST:.0f}.")
print("P-P means the pairing term is not involved in that collision at all.")
