"""What drives a wall-applied pair to 0.3022 nm, a distance the database never contains?

scripts/measure_type_pair_excluded_volume.py kills the obvious answer: the single sigma presses on
66 of 1923588 non-bonded pairs (0.0034 percent), so per-type sigma would buy almost nothing, and it
would make C4'-C4' WEAKER (0.3333 against 0.3975), not stronger. Yet after equilibration a pair with
|i-j| >= 3 was seen at 0.3022 nm, below the database's all-type minimum of 0.3333. So the question
is not the range. It is what walks a pair through a wall that at 0.3022 nm still delivers
4335 kJ/mol/nm.

This records the deepest |i-j| >= 3 approach of a run, then decomposes the force on those two beads
at that exact configuration, projected on the axis between them. Positive projection means the term
is pushing them apart; negative means it is the one driving them together. It also reports the
wall force with and without the 5000 cap, because a clipped wall is a constant force and no longer
depends on how much further in the pair goes.

Run: python scripts/diagnose_wall_penetration.py [ps] [nrep] [max_iter]
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
import cg_force_terms as FT           # noqa: E402

PS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MAXIT = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
MASS = 110.0
KB_INT = 0.008314462618
TARGET = 300.0
DT = 0.002
NSTEPS = int(round(PS / DT))
ATOM = ("P", "C4'", "N9/N1")

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"])
NB = 3 * L
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
xs = torch.tensor(s0["pos"].reshape(1, NB, 3), dtype=torch.float64)


def field(p, cap=None):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=cap)


x = xs.clone()
e_cur = float(field(x)[0].reshape(-1)[0])
s = 1e-5
resets = 0
for k in range(MAXIT):
    with torch.no_grad():
        _e, f = field(x)
        tr = x + s * f
        e_tr = float(field(tr)[0].reshape(-1)[0])
    if e_tr < e_cur:
        x, e_cur, s = tr, e_tr, min(s * 1.2, 1e-2)
    else:
        s *= 0.5
        if s < 1e-12:
            if resets >= 6:
                break
            resets += 1
            s = 1e-4
x_min = x.detach().clone()
with torch.no_grad():
    _e, fm = field(x_min)
print(f"{s0['name']}  {L} residues / {NB} beads / {len(ij)} WC pairs")
print(f"minimised {k+1} iters ({resets} restarts) to E = {e_cur:.2f}, "
      f"max|F| = {float(torch.linalg.norm(fm.reshape(-1, 3), dim=-1).max()):.2f}")
print()

xr = x_min.clone().repeat(NREP, 1, 1)
v = torch.zeros_like(xr)
temps = torch.full((NREP,), TARGET, dtype=torch.float64)
torch.manual_seed(99)


def ff(p):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=5000.0)[1]


best = (1e9, None)
for step in range(NSTEPS):
    with torch.no_grad():
        f = ff(xr)
        xr, v = C.batch_langevin_step(xr, v, f, temps, dt_ps=DT, mass_amu=MASS,
                                      friction=1.0, force_fn=ff)
        if step % 5 == 0:
            b = xr.reshape(NREP, NB, 3)
            d = torch.cdist(b, b) + torch.eye(NB)[None] * 1e6
            # pool the GLOBAL closest only if it happens to be |i-j| >= 3, which is how the first
            # version of this script found nothing: the global closest is nearly always a bonded
            # pair, so the wall-applied pairs were never even looked at. Mask first, then minimise.
            ix = torch.arange(NB)
            d[:, (ix[None, :] - ix[:, None]).abs() < 3] = 1e6
            dm, flat = d.reshape(NREP, -1).min(dim=1)
            for r in range(NREP):
                i, j = divmod(int(flat[r]), NB)
                if float(dm[r]) < best[0]:
                    best = (float(dm[r]), (step, r, i, j))

dm, info = best
if info is None:
    print("no |i-j| >= 3 pair came inside any recorded distance; nothing to diagnose")
    sys.exit(0)
step, r, i, j = info
print(f"deepest |i-j| >= 3 approach: {dm:.4f} nm at step {step} (replica {r}), "
      f"beads ({i},{j}) = {ATOM[i%3]}({i//3})-{ATOM[j%3]}({j//3}), |i-j| = {abs(i-j)}")
print(f"database all-type minimum at this gap: 0.3333 nm; C4'-C4' minimum 0.3333; "
      f"the shipped wall range 0.3975")
print()

one = xr[r:r+1].clone()
cl = C.GPUCellList(cell_size=1.5)
cl.build(one)
terms, energies = FT.term_energies_forces(one, ij, pw, cell_list=cl)
u = (one[0, i] - one[0, j])
u = u / torch.linalg.norm(u)
print("force on the pair, projected on the axis between them (+ pushes apart, - drives together):")
rows = []
for name, F in terms.items():
    proj = float((torch.as_tensor(F)[i] - torch.as_tensor(F)[j]) @ u)
    rows.append((abs(proj), proj, name))
rows.sort(reverse=True)
for mag, proj, name in rows:
    if mag > 1.0:
        print(f"    {name:22s} {proj:+11.2f} kJ/mol/nm")
tot = sum(p for _m, p, _n in rows)
print(f"    {'sum of the decomposed terms':22s} {tot:+11.2f} kJ/mol/nm")
print()

with torch.no_grad():
    _e_raw, f_raw = field(one, cap=None)
    _e_cap, f_cap = field(one, cap=5000.0)
clash_raw = float(torch.linalg.norm(f_raw[0, i] - f_raw[0, j]))
clash_cap = float(torch.linalg.norm(f_cap[0, i] - f_cap[0, j]))
true_clash = float(C._clash_pair_energy_force(
    (one[:, i] - one[:, j]).unsqueeze(0), torch.tensor([[dm]]), C.K_CLASH, C.CLASH_SIGMA)[1]
    .norm())
print(f"the wall's own force on this pair at {dm:.4f} nm:")
print(f"    uncapped, from the law      : {true_clash:11.2f} kJ/mol/nm")
print(f"    what cg_energy_forces returns with force_cap=None: {clash_raw:11.2f}")
print(f"    what cg_energy_forces returns with force_cap=5000: {clash_cap:11.2f}")
print()
print("If the uncapped total pushes the pair apart by far more than the sum of the other terms can")
print("close, the pair is not being held there by a force balance -- it got there through the wall")
print("in one step, and the question becomes the step size against the wall's stiffness, not the")
print("wall's range.")
