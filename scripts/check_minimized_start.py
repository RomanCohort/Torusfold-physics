"""Start from a minimum, as the pipeline does, instead of from the raw deposited structure.

Every collapse measurement in this session started from the deposited coordinates with v = 0 and
no minimisation. The pipeline does not do that: Level 1.5 and Level 2.5b call physical_relaxation,
and the REMD driver runs a relaxation phase before its main loop. So the measured "collapse" may
have been the unrelaxed starting structure, not the force field.

Minimisation uses the field's own force, x <- x + s*F, with s halved whenever the energy rises, so
it is monotone and follows exactly the same energy surface the dynamics uses (cap off, so the
descent is not clipped).

Run: python scripts/check_minimized_start.py [max_iter] [ps]
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

MAXIT = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
PS = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
MASS = 110.0
KB_INT = 0.008314462618
TARGET = 300.0
DT = 0.002
NSTEPS = int(round(PS / DT))
NREP = 2

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
x_start = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)


def field(p, cap=None):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=cap)


def rmsd(a, b):
    return float(torch.sqrt(((a - b) ** 2).sum(-1).mean()))


e0, f0 = field(x_start, cap=None)
print(f"{s0['name']} L={L} pairs={len(ij)}")
print(f"start:  E {float(e0.reshape(-1)[0]):12.2f} kJ/mol   max|F| "
      f"{float(torch.linalg.norm(f0.reshape(-1, 3), dim=-1).max()):9.2f} kJ/mol/nm")
print(f"        energy tensor requires_grad = {bool(e0.requires_grad)}  "
      f"(False => the caller cannot minimise through autograd, forces are the only handle)")
print()

x = x_start.clone()
e_cur = float(e0.reshape(-1)[0])
s = 1e-5
nacc = nrej = 0
for i in range(MAXIT):
    with torch.no_grad():
        _e, f = field(x, cap=None)
    trial = x + s * f
    with torch.no_grad():
        e_try = float(field(trial, cap=None)[0].reshape(-1)[0])
    if e_try < e_cur:
        x, e_cur, nacc = trial, e_try, nacc + 1
        s = min(s * 1.2, 1e-2)
    else:
        s *= 0.5
        nrej += 1
        if s < 1e-12:
            print("  step size exhausted")
            break
    if (i + 1) % max(1, MAXIT // 8) == 0:
        with torch.no_grad():
            _ee, ff = field(x, cap=None)
        print(f"  iter {i+1:5d}  E {e_cur:12.2f}  max|F| "
              f"{float(torch.linalg.norm(ff.reshape(-1, 3), dim=-1).max()):9.2f}  "
              f"RMSD from start {rmsd(x, x_start):.4f} nm  step {s:.2e}  "
              f"acc/rej {nacc}/{nrej}")
x_min = x.detach().clone()
with torch.no_grad():
    e_min, f_min = field(x_min, cap=None)
print()
print(f"minimised: E {float(e_min.reshape(-1)[0]):12.2f}   max|F| "
      f"{float(torch.linalg.norm(f_min.reshape(-1, 3), dim=-1).max()):9.2f}   "
      f"RMSD from start {rmsd(x_min, x_start):.4f} nm")
print(f"energy drop {float(e0.reshape(-1)[0]) - float(e_min.reshape(-1)[0]):.2f} kJ/mol")
print()

print(f"=== {PS} ps Langevin at {TARGET:.0f} K, {NREP} replicas, cap 5000, seed 99 ===")
print(f"{'start':12s} {'mean T (K)':>11s} {'T/tgt':>6s} {'closest med':>12s} "
      f"{'closest min':>12s} {'RMSD dr':>8s} {'E end':>10s}")
print("-" * 76)
keep = {}
for label, base in (("unminimised", x_start), ("minimised", x_min)):
    xr = base.clone().repeat(NREP, 1, 1)
    v = torch.zeros_like(xr)
    temps = torch.full((NREP,), TARGET, dtype=torch.float64)
    torch.manual_seed(99)

    def ff(p):
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(p)
        return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=5000.0)[1]

    Ts, dmins, edrift = [], [], []
    for step in range(NSTEPS):
        with torch.no_grad():
            f = ff(xr)
            xr, v = C.batch_langevin_step(xr, v, f, temps, dt_ps=DT, mass_amu=MASS,
                                          friction=1.0, force_fn=ff)
        if step % max(1, NSTEPS // 30) == 0 and step >= NSTEPS // 4:
            with torch.no_grad():
                Ts.append(float((MASS * (v ** 2).sum(dim=-1) / (3.0 * KB_INT)).mean()))
                b = xr.reshape(NREP, -1, 3)
                m = b.shape[1]
                d = torch.cdist(b, b) + torch.eye(m, device=b.device) * 10.0
                dmins.append(float(d.min()))
                edrift.append(float(C.cg_energy(xr, ij, pw).mean()))
    Ts = np.array(Ts); dmins = np.array(dmins); edrift = np.array(edrift)
    keep[label] = (Ts, dmins, edrift)
    drift = rmsd(xr[0].reshape(1, 3 * L, 3), base)
    print(f"{label:12s} {Ts.mean():11.1f} {Ts.mean()/TARGET:6.2f} "
          f"{np.median(dmins):12.4f} {dmins.min():12.4f} {drift:8.4f} {edrift[-1]:10.1f}")
print()
for label, (Ts, dmins, edrift) in keep.items():
    print(f"{label:12s} T over time: " + " ".join(f"{t:.0f}" for t in Ts))
    print(f"{label:12s} closest    : " + " ".join(f"{d:.3f}" for d in dmins))
print()
print("If the minimised start holds its temperature and its contacts while the unminimised one")
print("does not, the earlier collapse was the unrelaxed starting structure, not the force field.")
