"""Anneal safety probe: at the planned anneal temperature, is the chain leaving the native basin
by a physical route, or is it being held apart by the excluded-volume wall?

two_start_ergodicity.py builds "start B" by annealing the native 1L2X conformation at a fixed
temperature for ANNEAL_STEPS and taking one replica's final coordinates as the second start.
That is only a meaningful second start if the annealed conformation is still a conformation of
THIS chain in THIS field. Above some temperature the excluded-volume term stops being a
constraint and becomes the structure: the chain is a set of beads pushed apart by a k=20000
potential, its Rg is set by the balance of a saturated repulsion rather than by backbone and
stacking, and the two-start verdict then measures the anneal artifact instead of the sampler.

The quantity that decides this is not Rg alone -- a chain can open up physically or be blown
open. It is the clash share, E_clash / E_total: near zero the field is still doing the folding,
near one the wall is.

Per temperature, every CHECK_EVERY steps:
  Rg          nm, over all 3L CG sites (same definition as two_start_ergodicity.py)
  E_tot       kJ/mol, the full field with the BSJ trio zeroed
  E_clash     kJ/mol, the excluded-volume term alone
  clash %     E_clash / E_tot
  overlaps    bead pairs closer than CLASH_SIGMA, i.e. pairs the wall is actively separating
  max|F|      kJ/mol/nm, and "sat" = how many force components sit at >= 99% of force_cap.
              A saturated component is one the cap is clipping, i.e. F = -dE/dx no longer holds
              there. The cap is a guard rail, not a force law (README); a run that lives at the
              cap is a run the integrator is losing control of.
  displ       nm, mean per-site displacement from the native deposited conformation (internal
              change, not just rotation: the field is translation/rotation invariant)

300 K is carried as the control. A temperature whose clash share and saturation stay near the
300 K numbers is a physical perturbation; one where they climb by an order of magnitude is not,
whatever its Rg does.

This script only samples; it writes no checkpoint and decides nothing. Its output is the input
to the choice of ANNEAL_TEMP for the real run.

Run: python scripts/anneal_safety_probe.py [nsteps] [temp1,temp2,...]
     defaults: 2000 steps, 300,600,800,1000 K
"""
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B                    # noqa: E402
import torusfold.scheme2.torch_cgsim as C       # noqa: E402

NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
TEMPS = ([float(t) for t in sys.argv[2].split(",")] if len(sys.argv) > 2
         else [300.0, 600.0, 800.0, 1000.0])
CHECK_EVERY = 500
SEED = 20260218
FRICTION = 0.1
DT_PS = 0.002
MASS_AMU = 110.0
IDX = 0

# Same field as two_start_ergodicity.py: 1L2X is a LINEAR deposited chain, and the BSJ trio acts
# on P(0)-P(L-1), which a linear chain does not have. Leaving it on would measure the compaction
# artifact (1263-4159 kJ/mol/nm, attribute_compaction) instead of the anneal.
C.K_BSJ = 0.0
C.K_BSJ_GUIDE = 0.0
C.K_BSJ_CONTACT = 0.0

CAP = float(inspect.signature(C.cg_energy_forces).parameters["force_cap"].default)

pool = [s for s in B.load_structures(limit=400)
        if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[IDX]
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
native = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)


def _forces_at(p):
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl2)[1]


def _rg(p):
    c = p.mean(dim=1, keepdim=True)
    return float(torch.linalg.norm(p - c, dim=-1).pow(2).mean(dim=1).sqrt()[0])


def diagnose(pos):
    """Full-field energy, the clash term alone, and the geometry the wall is acting on."""
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    with torch.no_grad():
        e_tot, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
        e_cl, _ = C._explicit_forces_clash(pos, cl, C.K_CLASH, C.CLASH_SIGMA)
        _pi, _pj, _delta, dist = cl.get_pair_info(pos)
    n_ov = int((dist < C.CLASH_SIGMA).sum())
    fmax = float(f.abs().amax())
    sat = int((f.abs() >= 0.99 * CAP).sum())
    displ = float(torch.linalg.norm(pos[0] - native[0], dim=-1).mean())
    et, ec = float(e_tot[0]), float(e_cl[0])
    return dict(rg=_rg(pos), e_tot=et, e_clash=ec,
                share=(ec / et if et != 0 else float("nan")),
                n_ov=n_ov, fmax=fmax, sat=sat, displ=displ)


print(f"structure {s0['name']}  L={L}  pairs={len(ij)}")
print(f"field: K_BSJ=0, K_BSJ_GUIDE=0, K_BSJ_CONTACT=0 (linear chain, per README)")
print(f"integrator: dt={DT_PS} ps, mass {MASS_AMU} Da, friction {FRICTION}/ps, BAOAB, "
      f"force_cap={CAP:g}")
print(f"field: K_BB={C.K_BB}  K_CLASH={C.K_CLASH}  CLASH_SIGMA={C.CLASH_SIGMA}  "
      f"K_PAIR={C.K_PAIR}  K_ANGLE={C.K_ANGLE}  K_DIH={C.K_DIH}  K_STACK={C.K_STACK}")
print(f"the real run anneals {20000} steps; this probe runs {NSTEPS} per temperature and asks "
      f"whether the trend at each temperature is physical")
print()

d0 = diagnose(native)
print(f"native: Rg {d0['rg']:.3f} nm  E_tot {d0['e_tot']:.1f}  E_clash {d0['e_clash']:.1f} "
      f"({d0['share']:.1%})  overlaps {d0['n_ov']}  max|F| {d0['fmax']:.0f}  sat {d0['sat']}")
print()

HEAD = (f"{'step':>6s} {'Rg':>7s} {'E_tot':>11s} {'E_clash':>10s} {'clash%':>7s} "
        f"{'ovlp':>5s} {'max|F|':>8s} {'sat':>6s} {'displ':>7s}")
summary = []
for T in TEMPS:
    pos = native.repeat(1, 1, 1).clone()
    vel = torch.zeros_like(pos)
    temps = torch.full((1,), T, dtype=torch.float64)
    torch.manual_seed(SEED)          # identical noise sequence at every temperature
    print(f"=== {T:.0f} K ===")
    print(HEAD)
    print("-" * len(HEAD))
    t0 = time.time()
    last = d0
    for step in range(NSTEPS):
        with torch.no_grad():
            cl = C.GPUCellList(cell_size=1.5)
            cl.build(pos)
            _e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
            pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=DT_PS,
                                             mass_amu=MASS_AMU, friction=FRICTION,
                                             force_fn=_forces_at)
        if (step + 1) % CHECK_EVERY == 0:
            d = diagnose(pos)
            print(f"{step + 1:6d} {d['rg']:7.3f} {d['e_tot']:11.1f} {d['e_clash']:10.1f} "
                  f"{d['share']:7.1%} {d['n_ov']:5d} {d['fmax']:8.0f} {d['sat']:6d} "
                  f"{d['displ']:7.3f}")
            last = d
    print(f"  ({time.time() - t0:.0f} s)")
    print()
    summary.append((T, last))

print("=== summary: the LAST checkpoint of each temperature ===")
print(f"{'T (K)':>6s} {'Rg':>7s} {'clash%':>8s} {'ovlp':>5s} {'max|F|':>8s} {'sat':>6s} "
      f"{'displ':>7s}  {'verdict':s}")
print("-" * 78)
# The verdict is relative, and deliberately so: the question is not "is the clash share small"
# but "does raising the temperature change it by an order of magnitude". 300 K is the control,
# so its own row is the yardstick and is not judged.
base = summary[0][1] if summary and summary[0][0] == 300.0 else None
for T, d in summary:
    if base is None or T == 300.0:
        v = "(control)"
    elif d["share"] > max(10 * base["share"], 0.5):
        v = "NON-PHYSICAL: the wall is doing the work"
    elif d["sat"] > 10 * max(base["sat"], 1):
        v = "SUSPECT: force cap clipping everywhere"
    else:
        v = "physical"
    print(f"{T:6.0f} {d['rg']:7.3f} {d['share']:8.1%} {d['n_ov']:5d} {d['fmax']:8.0f} "
          f"{d['sat']:6d} {d['displ']:7.3f}  {v}")
