"""Is the E1 block-to-block spread a slow collapse rather than a sampling artefact?

E1 (1L2X, 40-200 ps, 8 blocks) gave a block J spread of 23.1 percent. E1b doubled the block
length (400 -> 800 frames each, 20 -> 40 ps) and the spread stayed at 22.4 percent. A statistical
spread would have fallen by about 1/sqrt(2). It did not, so the blocks are not samples of one
distribution -- something is still moving.

The per-coordinate columns say what: `angle` drifts 0.917 -> 0.867 and `stack` 1.016 -> 0.928
across E1b's eight blocks, both monotonically downward. angle is P-P-P and stack is the
P(i)-P(i+2) distance, so both shrinking means the backbone is tightening locally. That is the
shape of a collapse, and this repository has met one before (NOTES.md, "CG planar collapse",
fixed by injecting helical z into the initialisation and adding a z-restraint).

This measures it directly instead of inferring it from the marginals:

    Rg            radius of gyration, per block. Monotone decrease = the chain is compacting.
    WC N-N        the 16 Watson-Crick N-N distances: mean, and how many are still in contact.
                  If the fold is being lost, these open; if it is collapsing, they over-tighten.
    angle, stack  the two ratios that were drifting, so the correlation is visible rather than
                  assumed.

The native start is reported alongside, because "Rg went from X to Y" says nothing without it.

Run: python scripts/check_fold_drift.py [n_rep] [n_steps] [idx] [friction] [stride] [burn]
                                    [--blocks=N]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402


def _opt_int(name, default):
    pre = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(pre):
            return int(a[len(pre):])
    return default


NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
FRICTION = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 25
_burn_arg = int(sys.argv[6]) if len(sys.argv) > 6 else 0
burn = _burn_arg if _burn_arg > 0 else max(NSTEPS // 5, 1)
NB = max(_opt_int("blocks", 8), 1)
SEED = 20260218

z = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
TAB = {c: {"sigma": float(z[f"{c}__sigma"])} for c in B.COORDS}

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[IDX]
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
NN = lambda i: 3 * i + 2
PBEAD = torch.arange(L) * 3

pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
temps = torch.full((NREP,), 300.0, dtype=torch.float64)


def rg(p):
    """Radius of gyration per replica. p: (B, N, 3) nm."""
    c = p.mean(dim=1, keepdim=True)
    return torch.linalg.norm(p - c, dim=-1).pow(2).mean(dim=1).sqrt()


def contacts(p):
    """The 16 WC N-N distances, per replica: (B, P)."""
    return torch.linalg.norm(p[:, NN(ij[:, 0])] - p[:, NN(ij[:, 1])], dim=-1)


print(f"structure {s0['name']}  L={L}  pairs={len(ij)}")
print(f"{NREP} replicas, {NSTEPS} steps = {NSTEPS * 0.002:.1f} ps; burn {burn * 0.002:.1f} ps; "
      f"{NB} blocks; friction {FRICTION}/ps")
_rg0 = float(rg(pos)[0])
_c0 = contacts(pos)[0]
print(f"native start: Rg = {_rg0:.3f} nm,  WC N-N mean = {float(_c0.mean()):.3f} nm, "
      f"max = {float(_c0.max()):.3f} nm")
print()

b_rg = [[] for _ in range(NB)]
b_wc = [[] for _ in range(NB)]
b_acc = {b: {c: [0.0, 0.0, 0] for c in B.COORDS} for b in range(NB)}


def _forces_at(p):
    """Fresh forces at the post-update coordinates for the symplectic tail kick, so this run
    uses the same integrator as ibi_round0.py and the two sets of blocks are comparable."""
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl2)[1]


torch.manual_seed(SEED)
t0 = time.time()
for step in range(NSTEPS):
    with torch.no_grad():
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        _e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=0.002, mass_amu=110.0,
                                         friction=FRICTION, force_fn=_forces_at)
    if step >= burn and step % STRIDE == 0:
        with torch.no_grad():
            b = min(NB - 1, (step - burn) * NB // max(NSTEPS - burn, 1))
            b_rg[b].extend(rg(pos).tolist())
            b_wc[b].extend(contacts(pos).reshape(-1).tolist())
            for c in B.COORDS:
                q = B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64)
                b_acc[b][c][0] += q.sum()
                b_acc[b][c][1] += (q ** 2).sum()
                b_acc[b][c][2] += q.size

print(f"run: {time.time() - t0:.0f} s")
print()
print(f"{'block':>5s} {'window (ps)':>15s} {'Rg (nm)':>9s} {'dRg vs native':>14s} "
      f"{'WC mean':>8s} {'WC max':>7s} {'WC<1.5':>7s} {'angle':>6s} {'stack':>6s}")
print("-" * 92)
for b in range(NB):
    lo = burn + (NSTEPS - burn) * b // NB
    hi = burn + (NSTEPS - burn) * (b + 1) // NB
    if not b_rg[b]:
        print(f"{b + 1:5d} {lo * 0.002:6.1f}-{hi * 0.002:6.1f}   (no samples)")
        continue
    mr = float(np.mean(b_rg[b]))
    w = np.array(b_wc[b])
    ratios = {}
    for c in ("angle", "stack"):
        s1, s2, n = b_acc[b][c]
        ss = float(np.sqrt(max(s2 / n - (s1 / n) ** 2, 0.0)))
        ratios[c] = ss / TAB[c]["sigma"]
    print(f"{b + 1:5d} {lo * 0.002:6.1f}-{hi * 0.002:6.1f} {mr:9.3f} {mr - _rg0:+14.3f} "
          f"{w.mean():8.3f} {w.max():7.3f} {int((w < 1.5).sum()):7d} "
          f"{ratios['angle']:6.3f} {ratios['stack']:6.3f}")

print()
print(f"Rg: native {_rg0:.3f} nm -> block 1 {np.mean(b_rg[0]):.3f} -> "
      f"block {NB} {np.mean(b_rg[-1]):.3f} nm "
      f"({np.mean(b_rg[-1]) - np.mean(b_rg[0]):+.3f} nm across the window)")
print("A monotone Rg decrease alongside decreasing angle and stack means collapse, not sampling:")
print("the marginal keeps moving because the conformation keeps changing, and no run length")
print("converges it. If Rg is flat while the ratios drift, the cause is elsewhere.")
