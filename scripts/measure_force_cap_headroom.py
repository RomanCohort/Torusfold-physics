"""How much headroom does the force cap actually need?

The cap is applied to the summed force vector per bead, so it is not the gradient of any
potential: whatever it clips is a bias in the stationary distribution. It used to be the force
law -- 53.77 percent of beads sat exactly on it before the constants were recalibrated. It is now
3.08 percent, and the question is whether the remaining 200 kJ/mol/nm is above everything the
field actually produces or is still clipping real forces.

That is a measurement, not an opinion: run the field with force_cap=None and look at the
distribution of |F|. The cap can then be set above the physical maximum so that it catches only
numerical blow-ups, which is the only job a cap can do without biasing the ensemble.

Run: python scripts/measure_force_cap_headroom.py [n_structs] [n_steps]
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

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
CAP = 200.0

structs = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 4][:NS]
print(f"{len(structs)} native structures, full field, force_cap=None")
print()

mags = []
per_struct_max = []
for s in structs:
    L = len(s["pos"])
    pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
    ij = torch.tensor(s["pairs"], dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(ij), dtype=torch.float32)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    _e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl, force_cap=None)
    m = torch.linalg.norm(f.reshape(-1, 3), dim=-1).numpy()
    mags.append(m)
    per_struct_max.append(float(m.max()))
m = np.concatenate(mags)
print("=== |F| per bead at NATIVE geometry, cap off ===")
print(f"  beads {m.size}")
for q in (50, 90, 99, 99.9, 100):
    print(f"  p{q:<5} {np.percentile(m, q):10.2f} kJ/mol/nm")
print(f"  max over all structures   {m.max():10.2f}")
print(f"  beads above {CAP:.0f}           {int((m > CAP).sum())} "
      f"({100 * (m > CAP).mean():.4f}%)")
print(f"  beads above 400           {int((m > 400).sum())}")
print(f"  beads above 1000          {int((m > 1000).sum())}")
print(f"  worst single structure    {max(per_struct_max):10.2f}")
print()

# A short trajectory, because the cap's job is to survive the deformed configurations too.
s0 = structs[0]
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
vel = torch.zeros_like(pos)
temps = torch.full((1,), 300.0, dtype=torch.float64)
torch.manual_seed(20260220)


def _forces_at(p):
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl2, force_cap=None)[1]


traj = []
for step in range(NSTEPS):
    with torch.no_grad():
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        _e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=0.002,
                                         mass_amu=110.0, friction=1.0, force_fn=_forces_at)
    if step % 50 == 0:
        with torch.no_grad():
            cl = C.GPUCellList(cell_size=1.5)
            cl.build(pos)
            _e, fr = C.cg_energy_forces(pos, ij, pw, cell_list=cl, force_cap=None)
            traj.append(torch.linalg.norm(fr.reshape(-1, 3), dim=-1).numpy())
t = np.concatenate(traj)
print(f"=== |F| per bead over a {NSTEPS * 0.002:.0f} ps trajectory at 300 K, cap off ===")
print(f"  samples {t.size}")
for q in (50, 99, 99.9, 100):
    print(f"  p{q:<5} {np.percentile(t, q):10.2f} kJ/mol/nm")
print(f"  max                       {t.max():10.2f}")
print(f"  above {CAP:.0f}                 {int((t > CAP).sum())} "
      f"({100 * (t > CAP).mean():.4f}%)")
print(f"  above 400                 {int((t > 400).sum())}")
print(f"  above 1000                {int((t > 1000).sum())}")
