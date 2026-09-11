"""How often does the force cap fire on the current field? While it fires, F != -dE/dx."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C
import inspect

CAP = inspect.signature(C.cg_energy_forces).parameters["force_cap"].default
NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"])
pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(2, 1, 1)
vel = torch.zeros_like(pos)
temps = torch.full((2,), 300.0, dtype=torch.float64)
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float64)
print(f"cap = {CAP} kJ/mol/nm; {s0['name']}, 2 replicas, {NSTEPS * 0.002:.1f} ps")


def ff(p):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl)[1]


def uncapped(p):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=None)[1]


bad = 0
tot = 0
worst = 0.0
with torch.no_grad():
    for step in range(NSTEPS):
        f = ff(pos)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=0.002, mass_amu=110.0,
                                         friction=0.1, force_fn=ff)
        if step >= NSTEPS // 4 and step % 10 == 0:
            g = torch.linalg.norm(uncapped(pos).reshape(-1, 3), dim=-1)
            bad += int((g > CAP).sum())
            tot += g.numel()
            worst = max(worst, float(g.max()))
print(f"bead-force instances over the cap: {bad} of {tot} = {100.0 * bad / max(tot,1):.4f}%")
print(f"largest uncapped bead force seen   : {worst:.1f} kJ/mol/nm  ({worst / CAP:.2f}x the cap)")
